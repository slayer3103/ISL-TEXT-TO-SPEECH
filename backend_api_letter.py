#!/usr/bin/env python3
"""
backend_api.py - Flask backend for single-frame inference + optional audio caching.

Changes vs your previous version:
- Responses include `distribution` (label -> probability 0..1), `alternatives` (top-N)
- `confidence` is now a 0..1 float (not *100)
- Added /align endpoint for simple streaming alignment (greedy collapse + optional probability threshold)
- Blank and <UNK> handled explicitly (won't be artificially boosted)
- Audio caching unchanged (pyttsx3 background thread)
"""

import os
import base64
import json
import threading
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

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

AUDIO_CACHE_DIR = ROOT / "static" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "static"))
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Backend device:", device)

# Config
PROB_THRESH = float(os.environ.get("PROB_THRESH", 0.6))  # threshold in 0..1
USE_GUARD = True
TOP_K_ALTS = int(os.environ.get("TOP_K_ALTS", 4))

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

# ------------------ TorchScript optional models ------------------
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
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        time.sleep(0.05)
        return True, None
    except Exception as e:
        return False, str(e)

def _ensure_audio_cached(text: str):
    fname = _text_to_hash_filename(text)
    out_path = AUDIO_CACHE_DIR / fname
    url = f"/audio_cache/{fname}"
    if out_path.exists():
        return True, url

    def _job():
        ok, err = _synthesize_wav_pyttsx3(text, out_path)
        if not ok:
            print("TTS synth failed for text:", text, "err:", err)
    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return False, url

@app.route("/audio_cache/<path:fname>")
def serve_audio(fname):
    return send_from_directory(str(AUDIO_CACHE_DIR), fname)

# ------------------ Helpers for packaging distribution & alts ------------------
def topk_from_probs(probs: np.ndarray, labels_map: Optional[List[str]] = None, k: int = 4):
    """
    Given a 1D numpy probs array, return list of (label, prob) sorted desc top-k.
    If labels_map provided, index -> label using labels_map, else use str(idx).
    """
    if probs is None:
        return []
    idxs = np.argsort(probs)[::-1][:k]
    out = []
    for i in idxs:
        label = (labels_map[i] if labels_map and i < len(labels_map) else str(int(i)))
        prob = float(probs[i])
        out.append((label, prob))
    return out

def build_distribution_dict(probs: np.ndarray, labels_map: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Return dict label->prob (floats in 0..1). If labels_map provided, map indices.
    """
    if probs is None:
        return {}
    out = {}
    for i, p in enumerate(probs.tolist()):
        lab = labels_map[i] if labels_map and i < len(labels_map) else str(i)
        out[lab] = float(p)
    return out

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
    Returns JSON with normalized distribution and top alternatives:
    {
      "source": "landmark_mlp" | "word_ts" | "alpha_ts" | "guard",
      "label": "<LABEL>" or "<UNK>" or "<BLANK>",
      "confidence": 0..1,
      "distribution": {"a":0.7, "b":0.1, ...},
      "alternatives": [{"label":"x", "prob":0.1}, ...],
      "audio_url": "/audio_cache/...", "audio_ready": bool (if speak requested)
    }
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
        resp = {"source": "guard", "label": "<BLANK>", "confidence": 1.0, "distribution": {"<BLANK>": 1.0}, "alternatives": []}
        if data.get("speak", False):
            exists, url = _ensure_audio_cached("<BLANK>")
            resp["audio_url"] = url
            resp["audio_ready"] = exists
        return jsonify(resp)

    # Utility to postprocess probs -> response
    def build_resp_from_probs(probs: np.ndarray, labels_map: Optional[List[str]] = None, source_name: str = "model"):
        # ensure probs is 1D numpy array and normalized
        probs = np.array(probs, dtype=float)
        # stabilize tiny negative / numerical issues
        probs = np.clip(probs, 0.0, None)
        s = probs.sum()
        if s <= 0:
            # fallback uniform
            probs = np.ones_like(probs) / float(len(probs))
            s = 1.0
        probs = probs / s
        # top idx and value
        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        label_name = (labels_map[top_idx] if labels_map and top_idx < len(labels_map) else str(top_idx))
        out_label = label_name if top_prob >= PROB_THRESH else "<UNK>"

        distribution = build_distribution_dict(probs, labels_map)
        alts = topk_from_probs(probs, labels_map, k=TOP_K_ALTS)
        alternatives = [{"label": a, "prob": p} for (a, p) in alts if a != label_name]

        resp = {
            "source": source_name,
            "label": out_label,
            "confidence": float(top_prob),
            "distribution": distribution,
            "alternatives": alternatives
        }
        return resp

    # Try word TorchScript first (if available)
    if word_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = word_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            resp = build_resp_from_probs(probs, word_labels, source_name="word_ts")
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(resp["label"])
                resp["audio_url"] = url
                resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_ts inference error:", e)

    # Try alphabet TS
    if alpha_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = alpha_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            resp = build_resp_from_probs(probs, alpha_labels, source_name="alpha_ts")
            # map indices to inv_labels if alpha_labels missing
            if inv_labels and (resp["label"] == "<UNK>" or resp["label"].startswith("<")):
                # try to map top index to inv_labels
                top_idx = int(np.argmax(probs))
                mapped = inv_labels.get(top_idx)
                if mapped:
                    resp["label"] = mapped if float(np.max(probs)) >= PROB_THRESH else "<UNK>"
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(resp["label"])
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
        resp = build_resp_from_probs(probs, list(inv_labels.values()) if inv_labels else None, source_name="landmark_mlp")
        if data.get("speak", False):
            exists, url = _ensure_audio_cached(resp["label"])
            resp["audio_url"] = url
            resp["audio_ready"] = exists
        return jsonify(resp)
    except Exception as e:
        print("Model inference error:", e)
        return jsonify({"error": "model inference error", "detail": str(e)}), 500

@app.route("/align", methods=["POST"])
def align():
    """
    Simple server-side streaming alignment/collapse endpoint.

    Accepts:
      {"frames": [ {"label": "a", "prob": 0.75, "timestamp": 123}, ... ],
       "blank_tokens": ["<BLANK>", "<UNK>"], 
       "min_prob": 0.0
      }

    Returns:
      {
        "aligned": "this is",
        "segments": [ {"label":"t","start_idx":0,"end_idx":1,"avg_prob":0.8}, ... ],
        "collapsed": ["t","h","i","s"]
      }

    This is a greedy collapse: it removes repeated consecutive identical labels and removes any label
    present in blank_tokens. It can optionally filter out labels whose average prob < min_prob.
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    frames = data.get("frames", [])
    blank_tokens = data.get("blank_tokens", ["<BLANK>", "<UNK>"])
    min_prob = float(data.get("min_prob", 0.0))

    if not frames:
        return jsonify({"aligned": "", "segments": [], "collapsed": []})

    # frames: list of dicts with keys label and prob (0..1)
    collapsed = []
    segments = []
    last_label = None
    seg_start = 0
    seg_probs = []

    for i, f in enumerate(frames):
        lbl = f.get("label")
        prob = float(f.get("prob", 0.0))
        # treat blanks/unk as skip
        if lbl in blank_tokens:
            # finalize any existing segment
            if last_label is not None:
                avg = float(np.mean(seg_probs)) if seg_probs else 0.0
                if avg >= min_prob:
                    segments.append({"label": last_label, "start_idx": seg_start, "end_idx": i-1, "avg_prob": avg})
                    collapsed.append(last_label)
                # reset
                last_label = None
                seg_probs = []
            continue

        # if same as previous, accumulate
        if lbl == last_label:
            seg_probs.append(prob)
            continue
        # new label encountered
        if last_label is not None:
            avg = float(np.mean(seg_probs)) if seg_probs else 0.0
            if avg >= min_prob:
                segments.append({"label": last_label, "start_idx": seg_start, "end_idx": i-1, "avg_prob": avg})
                collapsed.append(last_label)
        # start new segment
        last_label = lbl
        seg_start = i
        seg_probs = [prob]

    # finalize last segment
    if last_label is not None:
        avg = float(np.mean(seg_probs)) if seg_probs else 0.0
        if avg >= min_prob:
            segments.append({"label": last_label, "start_idx": seg_start, "end_idx": len(frames)-1, "avg_prob": avg})
            collapsed.append(last_label)

    # join collapsed into a string with spaces where appropriate (we keep labels as-is)
    aligned = " ".join(collapsed) if collapsed else ""

    return jsonify({"aligned": aligned, "segments": segments, "collapsed": collapsed})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
