#!/usr/bin/env python3
"""
backend_api.py - Flask backend for single-frame inference + optional audio caching.

Improvements added:
- Guard: if both hands absent -> return {"label":"<BLANK>"}
- Probability thresholding: low-confidence -> {"label":"<UNK>"}
- Audio caching: synthesize label audio using pyttsx3 into static/audio_cache/<sha>.wav
  - If cached audio exists, return audio_url immediately.
  - If not, spawn background thread to create WAV and return audio_pending.
- Mediapipe processing is thread-safe via a lock and recreate-on-error strategy.

Run:
  python backend_api.py
"""

import os
import base64
import json
import threading
import hashlib
import time
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
import torch
import mediapipe as mp
import joblib
import pyttsx3

# Import helper functions from app_isl
from app_isl import build_model, extract_lr_landmarks, load_model_and_scaler

ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "models"
LANDMARK_MODEL_DIR = MODEL_DIR / "landmark_model"

# audio cache folder (served as static)
AUDIO_CACHE_DIR = ROOT / "static" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "static"))
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Backend device:", device)

# Config
PROB_THRESH = float(os.environ.get("PROB_THRESH", 0.6))
USE_GUARD = True

# ------------------ Load landmark model + scaler ------------------
LANDMARK_MODEL_PATH = LANDMARK_MODEL_DIR / "final_model.pt"
SCALER_PATH = LANDMARK_MODEL_DIR / "scaler_and_map.joblib"

landmark_model = None
mean_arr = None
std_arr = None
inv_labels = None

if LANDMARK_MODEL_PATH.exists() and SCALER_PATH.exists():
    try:
        model_state, mean_arr, std_arr, labels_map = load_model_and_scaler(str(LANDMARK_MODEL_PATH), str(SCALER_PATH), device)
        mean_arr = np.array(mean_arr, dtype=np.float32) if mean_arr is not None else None
        std_arr = np.array(std_arr, dtype=np.float32) if std_arr is not None else None
        num_classes = len(labels_map)
        landmark_model = build_model(84, [256, 128], num_classes, device)
        landmark_model.load_state_dict(model_state)
        landmark_model.to(device)
        landmark_model.eval()
        inv_labels = {v: k for k, v in labels_map.items()}
        print("Loaded landmark MLP model with", num_classes, "classes.")
    except Exception as e:
        print("Error loading landmark model/scaler:", e)
else:
    print("Warning: landmark model or scaler not found at", LANDMARK_MODEL_PATH, SCALER_PATH)

# ------------------ TorchScript optional models (kept as before) ------------------
WORD_TS_PATH = MODEL_DIR / "word_model" / "export" / "model_ts.pt"
ALPHA_TS_PATH = MODEL_DIR / "alphabet_model" / "export" / "model_ts.pt"

alpha_ts = None
alpha_labels = None
word_ts = None
word_labels = None

if ALPHA_TS_PATH.exists():
    try:
        alpha_ts = torch.jit.load(str(ALPHA_TS_PATH), map_location=device).eval()
        lpath = ALPHA_TS_PATH.parent / "labels_map.json"
        if lpath.exists():
            alpha_labels = json.load(open(lpath, "r", encoding="utf-8"))
        print("Loaded alphabet TorchScript model:", ALPHA_TS_PATH)
    except Exception as e:
        print("Failed to load alphabet TS model:", e)

if WORD_TS_PATH.exists():
    try:
        word_ts = torch.jit.load(str(WORD_TS_PATH), map_location=device).eval()
        lpath = WORD_TS_PATH.parent / "labels_map.json"
        if lpath.exists():
            word_labels = json.load(open(lpath, "r", encoding="utf-8"))
        print("Loaded word TorchScript model:", WORD_TS_PATH)
    except Exception as e:
        print("Failed to load word TS model:", e)

# ------------------ Mediapipe Hands (static image mode) and lock ------------------
mp_h = mp.solutions.hands
_HANDS_KW = dict(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_h.Hands(**_HANDS_KW)
_mp_lock = threading.Lock()

def _process_frame_safe(rgb_image):
    global hands
    with _mp_lock:
        try:
            return hands.process(rgb_image)
        except ValueError as e:
            msg = str(e)
            if "Packet timestamp mismatch" in msg or "Graph has errors" in msg:
                try:
                    hands.close()
                except Exception:
                    pass
                hands = mp_h.Hands(**_HANDS_KW)
                return hands.process(rgb_image)
            raise

def _decode_b64_frame(b64_str: str):
    try:
        header, data = (b64_str.split(",", 1) if "," in b64_str else ("", b64_str))
        img_bytes = base64.b64decode(data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print("decode error:", e)
        return None

# ------------------ Audio utils: caching + background synthesis ------------------
def _text_to_hash_filename(text: str):
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{h}.wav"

def _synthesize_wav_pyttsx3(text: str, out_path: Path):
    """
    Blocking call to synthesize text->wav using pyttsx3.
    We'll call this inside a background thread.
    """
    try:
        engine = pyttsx3.init()
        # set properties if needed e.g. rate, volume, voice
        engine.setProperty('rate', 160)
        # Save to file
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        # small sleep to ensure file is flushed
        time.sleep(0.1)
        return True, None
    except Exception as e:
        return False, str(e)

def _ensure_audio_cached(text: str):
    """
    Ensure an audio file exists for 'text'. Returns (exists_now, audio_path).
    If not exists, spawn a background thread to create it and return (False, expected_path).
    """
    fname = _text_to_hash_filename(text)
    out_path = AUDIO_CACHE_DIR / fname
    url = f"/audio_cache/{fname}"
    if out_path.exists():
        return True, url

    # spawn background thread to synthesize
    def _job():
        ok, err = _synthesize_wav_pyttsx3(text, out_path)
        if not ok:
            print("TTS synth failed for text:", text, "err:", err)
    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return False, url

# Serve cached audio files from static/audio_cache
@app.route("/audio_cache/<path:fname>")
def serve_audio(fname):
    return send_from_directory(str(AUDIO_CACHE_DIR), fname)

# ------------------ API endpoints ------------------
@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "device": str(device),
        "has_landmark_model": bool(landmark_model is not None),
        "has_alpha_ts": bool(alpha_ts is not None),
        "has_word_ts": bool(word_ts is not None),
    })

@app.route("/infer", methods=["POST"])
def infer():
    """
    Expects JSON: {"frame": "data:image/jpeg;base64,....", "speak": true|false}
    Returns JSON: {"source": "...", "label": "...", "confidence": float, "audio_url": "/audio_cache/..." (if available), "audio_pending": bool}
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    if not data or "frame" not in data:
        return jsonify({"error": "no frame provided"}), 400

    img = _decode_b64_frame(data["frame"])
    if img is None:
        return jsonify({"error": "could not decode frame"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        results = _process_frame_safe(rgb)
    except Exception as e:
        print("MediaPipe processing error:", e)
        return jsonify({"error": "mediapipe processing error", "detail": str(e)}), 500

    # Extract features and presence
    try:
        feat, left_p, right_p = extract_lr_landmarks(results)  # feat shape (84,)
    except Exception as e:
        print("Feature extraction error:", e)
        return jsonify({"error": "feature extraction error", "detail": str(e)}), 500

    # Guard: if both hands absent: return blank immediately (fast)
    if USE_GUARD and (left_p == 0 and right_p == 0):
        resp = {"source": "guard", "label": "<BLANK>", "confidence": 1.0}
        # Optionally trigger audio caching (if client asked for speech)
        if data.get("speak", False):
            exists, url = _ensure_audio_cached("<BLANK>")
            resp["audio_url"] = url
            resp["audio_ready"] = exists
        return jsonify(resp)

    # Try word TorchScript first (if available)
    if word_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = word_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = word_labels[idx] if word_labels and idx < len(word_labels) else str(idx)
            # thresholding
            if prob < PROB_THRESH:
                out_label = "<UNK>"
            else:
                out_label = label
            resp = {"source": "landmark_mlp", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url
                resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            # fallback to other models
            print("word_ts inference error:", e)

    # Try alphabet TS
    if alpha_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = alpha_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            prob = float(probs[idx])
            label = alpha_labels[idx] if alpha_labels and idx < len(alpha_labels) else (inv_labels.get(idx, str(idx)) if inv_labels else str(idx))
            out_label = label if prob >= PROB_THRESH else "<UNK>"
            resp = {"source": "landmark_mlp", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url
                resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("alpha_ts inference error:", e)

    # Fallback to landmark MLP
    if landmark_model is None or mean_arr is None or std_arr is None:
        return jsonify({"error": "no landmark model available"}), 500

    try:
        x = (feat - mean_arr) / std_arr
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = landmark_model(x_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        prob = float(probs[idx])
        label = inv_labels.get(idx, str(idx)) if inv_labels else str(idx)
        out_label = label if prob >= PROB_THRESH else "<UNK>"
        resp = {"source": "landmark_mlp", "label": out_label, "confidence": float(prob * 100)}
        if data.get("speak", False):
            exists, url = _ensure_audio_cached(out_label)
            resp["audio_url"] = url
            resp["audio_ready"] = exists
        return jsonify(resp)
    except Exception as e:
        print("Model inference error:", e)
        return jsonify({"error": "model inference error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
