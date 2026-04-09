#!/usr/bin/env python3
"""
backend_api_multi.py - Flask backend that supports:
 - single-frame inference (alphabet/landmark MLP / TorchScript models)
 - sequence (multi-frame) inference for word LSTM (endpoint /infer_seq)
 - audio caching (pyttsx3)
 - mediapipe thread-safe processing
 - serves frontend from static/frontend

Place this file in your project root. Expects:
 - models/landmark_model/... (final_model.pt, scaler joblib)
 - models/alphabet_model/export/model_ts.pt (+ labels_map.json) [optional]
 - models/word_model/export/model_ts.pt (+ labels_map.json) [optional]
 - models/word_model/best.pt (LSTM checkpoint with "model_state" and "labels") [optional]

Endpoints:
 - GET  /health
 - POST /infer      -> single-frame behaviour (existing infer)
 - POST /infer_seq  -> accepts {"frames": [...base64...], "speak": bool, "thresh": float}
 - GET  /audio_cache/<fname>
 - GET  /           -> serves frontend index.html (if present)
"""

import os
import base64
import json
import threading
import hashlib
import time
from pathlib import Path
from typing import Optional, List, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
import torch
import mediapipe as mp
import joblib
import pyttsx3

# Import helper functions from app_isl if available
try:
    from app_isl import build_model as build_landmark_mlp, extract_lr_landmarks, load_model_and_scaler
except Exception:
    # Minimal fallback if app_isl not importable
    def extract_lr_landmarks(results):
        left = [0.0] * 42
        right = [0.0] * 42
        left_p = 0
        right_p = 0
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label.lower()
                pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                flat = [v for pair in pts for v in pair]
                if len(flat) == 42:
                    if label == "left":
                        left = flat; left_p = 1
                    elif label == "right":
                        right = flat; right_p = 1
        feat = np.array(left + right, dtype=np.float32)
        return feat, left_p, right_p

    def load_model_and_scaler(model_path, scaler_path, device):
        ck = torch.load(model_path, map_location="cpu")
        model_state = ck.get("model_state", ck)
        mean = ck.get("mean", None)
        std = ck.get("std", None)
        labels_map = ck.get("labels_map", None)
        if labels_map is None or mean is None or std is None:
            info = joblib.load(scaler_path)
            if labels_map is None:
                labels_map = info.get("labels_map")
            if mean is None:
                mean = info.get("mean")
            if std is None:
                std = info.get("std")
        return model_state, np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32), labels_map

    def build_landmark_mlp(input_dim, hidden_dims, num_classes, device, dropout=0.4):
        import torch.nn as nn
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                layers = []
                prev = input_dim
                for h in hidden_dims:
                    layers.append(nn.Linear(prev, h))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    prev = h
                layers.append(nn.Linear(prev, num_classes))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x)
        model = MLP().to(device)
        return model

ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "models"
LANDMARK_MODEL_DIR = MODEL_DIR / "landmark_model"

# audio cache folder (served as static)
AUDIO_CACHE_DIR = ROOT / "static" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# static frontend folder (place your built frontend here)
FRONTEND_DIR = ROOT / "static" / "frontend"

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
        landmark_model = build_landmark_mlp(84, [256, 128], num_classes, device)
        landmark_model.load_state_dict(model_state)
        landmark_model.to(device)
        landmark_model.eval()
        inv_labels = {v: k for k, v in labels_map.items()}
        print("Loaded landmark MLP model with", num_classes, "classes.")
    except Exception as e:
        print("Error loading landmark model/scaler:", e)
else:
    print("Warning: landmark model or scaler not found at", LANDMARK_MODEL_PATH, SCALER_PATH)

# ------------------ TorchScript optional models (alphabet + word) ------------------
WORD_TS_PATH = MODEL_DIR / "word_model" / "export" / "model_ts.pt"
ALPHA_TS_PATH = MODEL_DIR / "alphabet_model" / "export" / "model_ts.pt"

alpha_ts = None
alpha_labels = None
word_ts = None
word_ts_labels = None

def try_load_ts(path):
    try:
        m = torch.jit.load(str(path), map_location=device).eval()
        return m
    except Exception as e:
        print("Failed to load torchscript model at", path, ":", e)
        return None

if ALPHA_TS_PATH.exists():
    alpha_ts = try_load_ts(ALPHA_TS_PATH)
    lpath = ALPHA_TS_PATH.parent / "labels_map.json"
    if lpath.exists():
        alpha_labels = json.load(open(lpath, "r", encoding="utf-8"))
    if alpha_ts:
        print("Loaded alphabet TorchScript model:", ALPHA_TS_PATH)

if WORD_TS_PATH.exists():
    word_ts = try_load_ts(WORD_TS_PATH)
    lpath = WORD_TS_PATH.parent / "labels_map.json"
    if lpath.exists():
        word_ts_labels = json.load(open(lpath, "r", encoding="utf-8"))
    if word_ts:
        print("Loaded word TorchScript model:", WORD_TS_PATH)

# ------------------ Load LSTM word model checkpoint (non-script) if present ------------------
WORD_LSTM_CKPT = MODEL_DIR / "word_model" / "best.pt"
word_lstm_model = None
word_lstm_labels = None
word_lstm_device = device

def build_word_lstm(input_dim=109, hidden=256, nlayers=2, nclass=10, dropout=0.3, device=device):
    import torch.nn as nn
    class BiLSTM(nn.Module):
        def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2,128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128,nclass))
        def forward(self,x,mask):
            out,_ = self.lstm(x)
            maskf = mask.unsqueeze(-1).float()
            pooled = (out*maskf).sum(1) / maskf.sum(1).clamp(min=1e-3)
            return self.head(pooled)
    dev = torch.device("cuda" if (device.type == "cuda" and torch.cuda.is_available()) else "cpu")
    model = BiLSTM(input_dim, hidden, nlayers, nclass, dropout).to(dev)
    return model, dev

if WORD_LSTM_CKPT.exists():
    try:
        ck = torch.load(str(WORD_LSTM_CKPT), map_location="cpu")
        state = ck.get("model_state", ck)
        labels = ck.get("labels") or ck.get("labels_map")
        if labels is None:
            print("Warning: word LSTM checkpoint has no labels key.")
            word_lstm_labels = None
        else:
            word_lstm_labels = list(labels)
        nclass = len(word_lstm_labels) if word_lstm_labels else (state["head.4.weight"].shape[0] if "head.4.weight" in state else 100)
        word_lstm_model, word_lstm_device = build_word_lstm(input_dim=109, nclass=nclass, device=device)
        word_lstm_model.load_state_dict(state)
        word_lstm_model.to(word_lstm_device)
        word_lstm_model.eval()
        print("Loaded word LSTM checkpoint with", nclass, "classes.")
    except Exception as e:
        print("Failed loading word LSTM checkpoint:", e)

# ------------------ Mediapipe Hands / Holistic and lock ------------------
mp_hol = mp.solutions.holistic
mp_h = mp.solutions.hands
_HANDS_KW = dict(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
# For sequence inference we will use Holistic (pose + hands). For single-frame hand-only use Hands.
hol = mp_hol.Holistic(static_image_mode=True, model_complexity=0, refine_face_landmarks=False,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_h.Hands(**_HANDS_KW)
_mp_lock = threading.Lock()

def _process_frame_safe_hands(rgb_image):
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

def _process_frame_safe_holistic(rgb_image):
    global hol
    with _mp_lock:
        try:
            return hol.process(rgb_image)
        except ValueError as e:
            msg = str(e)
            if "Packet timestamp mismatch" in msg or "Graph has errors" in msg:
                try:
                    hol.close()
                except Exception:
                    pass
                hol = mp_hol.Holistic(static_image_mode=True, model_complexity=0,
                                      refine_face_landmarks=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
                return hol.process(rgb_image)
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

# ------------------ Audio utils (same as before) ------------------
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

# ------------------ Sequence feature extraction helpers (for word LSTM) ------------------
# We mimic the feature vector creation in word/8_app_words_infer_fast.py:
# feat = [left(42), right(42), pose(18), lw_vel(2), rw_vel(2), [dist], present(2)] => total 109

POSE_IDX = [0,11,12,13,14,15,16,23,24]  # same as word script, indices from pose landmarks

def extract_from_holistic(results) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return left(42), right(42), pose(18), present(np.array([left_p,right_p]))"""
    left = np.zeros(42, dtype=np.float32)
    right = np.zeros(42, dtype=np.float32)
    left_p = 0; right_p = 0
    if getattr(results, "left_hand_landmarks", None):
        left_p = 1
        left = np.array([float(v) for lm in results.left_hand_landmarks.landmark for v in (lm.x, lm.y)], dtype=np.float32)
    if getattr(results, "right_hand_landmarks", None):
        right_p = 1
        right = np.array([float(v) for lm in results.right_hand_landmarks.landmark for v in (lm.x, lm.y)], dtype=np.float32)
    pose = []
    if getattr(results, "pose_landmarks", None):
        for i in POSE_IDX:
            lm = results.pose_landmarks.landmark[i]
            pose.extend([float(lm.x), float(lm.y)])
    else:
        pose = [0.0]*(len(POSE_IDX)*2)
    present = np.array([left_p, right_p], dtype=np.int8)
    return left, right, np.array(pose, dtype=np.float32), present

def make_feat(left, right, pose, present, lw_vel, rw_vel):
    dist = 0.0
    if left.size and right.size:
        dist = float(np.linalg.norm(left[:2]-right[:2]))
    lw_v = lw_vel if lw_vel is not None else np.zeros(2, np.float32)
    rw_v = rw_vel if rw_vel is not None else np.zeros(2, np.float32)
    return np.concatenate([
        left if left.size else np.zeros(42, np.float32),
        right if right.size else np.zeros(42, np.float32),
        pose if pose.size else np.zeros(18, np.float32),
        lw_v.astype(np.float32),
        rw_v.astype(np.float32),
        np.array([dist], np.float32),
        present.astype(np.float32)
    ], axis=0)

# ------------------ API endpoints ------------------
@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "device": str(device),
        "has_landmark_model": bool(landmark_model is not None),
        "has_alpha_ts": bool(alpha_ts is not None),
        "has_word_ts": bool(word_ts is not None),
        "has_word_lstm": bool(word_lstm_model is not None),
    })

@app.route("/")
def index():
    # serve frontend index if exists
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({"ok": True, "note": "no frontend found in static/frontend. Place your files there."})

@app.route("/infer", methods=["POST"])
def infer():
    """
    Single-frame inference endpoint (backwards-compatible with previous /infer):
    JSON input: {"frame": "data:image/jpeg;base64,...", "speak": true/false, "thresh": 0.6}
    Response:
      {"source": "...", "label": "...", "confidence": float, "audio_url": "/audio_cache/...", "audio_ready": bool}
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    if not data or "frame" not in data:
        return jsonify({"error": "no frame provided"}), 400

    thresh = float(data.get("thresh", PROB_THRESH))

    img = _decode_b64_frame(data["frame"])
    if img is None:
        return jsonify({"error": "could not decode frame"}), 400

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Prefer the hands-only processing for speed
    try:
        results = _process_frame_safe_hands(rgb)
    except Exception as e:
        print("MediaPipe processing error:", e)
        return jsonify({"error": "mediapipe processing error", "detail": str(e)}), 500

    try:
        feat, left_p, right_p = extract_lr_landmarks(results)
    except Exception as e:
        print("Feature extraction error:", e)
        return jsonify({"error": "feature extraction error", "detail": str(e)}), 500

    if USE_GUARD and (left_p == 0 and right_p == 0):
        resp = {"source": "guard", "label": "<BLANK>", "confidence": 1.0}
        if data.get("speak", False):
            exists, url = _ensure_audio_cached("<BLANK>")
            resp["audio_url"] = url
            resp["audio_ready"] = exists
        return jsonify(resp)

    # Try word TorchScript first (if available) - expects single-frame feat shaped (D,)
    if word_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = word_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs)); prob = float(probs[idx])
            label = word_ts_labels[idx] if word_ts_labels and idx < len(word_ts_labels) else str(idx)
            out_label = label if prob >= thresh else "<UNK>"
            resp = {"source": "word_ts", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_ts inference error:", e)

    # Try alphabet TorchScript (if available)
    if alpha_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = alpha_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs)); prob = float(probs[idx])
            label = alpha_labels[idx] if alpha_labels and idx < len(alpha_labels) else (inv_labels.get(idx, str(idx)) if inv_labels else str(idx))
            out_label = label if prob >= thresh else "<UNK>"
            resp = {"source": "alpha_ts", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
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
        idx = int(np.argmax(probs)); prob = float(probs[idx])
        label = inv_labels.get(idx, str(idx)) if inv_labels else str(idx)
        out_label = label if prob >= thresh else "<UNK>"
        resp = {"source": "landmark_mlp", "label": out_label, "confidence": float(prob * 100)}
        if data.get("speak", False):
            exists, url = _ensure_audio_cached(out_label)
            resp["audio_url"] = url; resp["audio_ready"] = exists
        return jsonify(resp)
    except Exception as e:
        print("Model inference error:", e)
        return jsonify({"error": "model inference error", "detail": str(e)}), 500

@app.route("/infer_seq", methods=["POST"])
def infer_seq():
    """
    Sequence (multi-frame) inference for word LSTM.
    Input JSON:
      {"frames": ["data:image/jpeg;base64,...", ...], "speak": bool, "thresh": 0.6}
    - frames should be in chronological order (oldest -> newest).
    Returns:
      {"source": "word_lstm"|"word_ts"|"none", "label": "...", "confidence": float, "audio_ready": bool, "audio_url": "..."}
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    frames = data.get("frames")
    if not frames or not isinstance(frames, list):
        return jsonify({"error": "frames must be a non-empty list of base64 images"}), 400

    thresh = float(data.get("thresh", PROB_THRESH))

    # First try LSTM checkpoint model if loaded
    if word_lstm_model is not None:
        feats = []
        presents = []
        # we'll compute simple velocities based on consecutive frames
        prev_l = None; prev_r = None
        fps_approx = 30.0  # reasonable default
        for b64 in frames:
            img = _decode_b64_frame(b64)
            if img is None:
                # skip frames that fail
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            try:
                res = _process_frame_safe_holistic(rgb)
            except Exception as e:
                print("holistic processing error:", e)
                continue
            left, right, pose, present = extract_from_holistic(res)
            # compute velocities (simple difference * fps)
            lw = left[:2] if left is not None else None
            rw = right[:2] if right is not None else None
            lw_vel = np.zeros(2, np.float32)
            rw_vel = np.zeros(2, np.float32)
            if prev_l is not None and lw is not None:
                lw_vel = (lw - prev_l) * fps_approx
            if prev_r is not None and rw is not None:
                rw_vel = (rw - prev_r) * fps_approx
            prev_l = lw if lw is not None else prev_l
            prev_r = rw if rw is not None else prev_r
            feat = make_feat(left, right, pose, present, lw_vel, rw_vel)
            feats.append(feat)
            presents.append(1 if present.sum() > 0 else 0)

        if len(feats) == 0:
            return jsonify({"error": "no valid frames processed"}), 400

        X = np.stack(feats, axis=0)  # (T,D)
        M = np.array(presents, dtype=np.uint8)
        xt = torch.from_numpy(X).unsqueeze(0).to(word_lstm_device)
        mt = torch.from_numpy(M).unsqueeze(0).to(word_lstm_device)
        try:
            with torch.inference_mode():
                logits = word_lstm_model(xt, mt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax()); prob = float(probs[idx])
            label = word_lstm_labels[idx] if word_lstm_labels and idx < len(word_lstm_labels) else str(idx)
            out_label = label if prob >= thresh else "<UNK>"
            resp = {"source": "word_lstm", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_lstm inference error:", e)
            # Fallthrough to try torchscript if available

    # Fallback: if word TorchScript exists, try to create a single-frame representative (last frame) inference
    if word_ts is not None:
        last = frames[-1]
        img = _decode_b64_frame(last)
        if img is None:
            return jsonify({"error": "could not decode last frame for fallback"}), 400
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = _process_frame_safe_hands(rgb)
            feat, left_p, right_p = extract_lr_landmarks(results)
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = word_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs)); prob = float(probs[idx])
            label = word_ts_labels[idx] if word_ts_labels and idx < len(word_ts_labels) else str(idx)
            out_label = label if prob >= thresh else "<UNK>"
            resp = {"source": "word_ts_fallback", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_ts fallback error:", e)

    return jsonify({"error": "no sequence model available"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
