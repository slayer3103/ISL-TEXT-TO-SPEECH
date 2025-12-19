# backend_api.py
"""
Flask API for single-frame inference from the React frontend.

- Uses MediaPipe in static_image_mode=True to avoid packet timestamp mismatch.
- Protects calls to hands.process with a threading.Lock to avoid concurrent graph issues.
- Loads landmark MLP model (trained) and optionally TorchScript exports if present.
- Exposes /health and /infer endpoints.

Run:
  .\.venv\Scripts\Activate    # Windows PowerShell
  python backend_api.py
"""

import os
import base64
import json
import threading
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import cv2
import torch
import mediapipe as mp
import joblib

# Import helper functions from your existing script
# Ensure app_isl.py is in the project root and exposes these functions:
#   - build_model(input_dim, hidden_dims, num_classes, device, dropout=0.4)
#   - extract_lr_landmarks(results) -> (feat_array_84d, left_present, right_present)
#   - load_model_and_scaler(model_path, scaler_path, device) -> (model_state, mean, std, labels_map)
from app_isl import build_model, extract_lr_landmarks, load_model_and_scaler

ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "models"
LANDMARK_MODEL_DIR = MODEL_DIR / "landmark_model"
WORD_TS_PATH = MODEL_DIR / "word_model" / "export" / "model_ts.pt"
ALPHA_TS_PATH = MODEL_DIR / "alphabet_model" / "export" / "model_ts.pt"

app = Flask(__name__)
CORS(app)  # allow requests from frontend dev server (adjust in production)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Backend device:", device)

# ------------------ Load trained landmark MLP model + scaler ------------------
LANDMARK_MODEL_PATH = LANDMARK_MODEL_DIR / "final_model.pt"
SCALER_PATH = LANDMARK_MODEL_DIR / "scaler_and_map.joblib"

landmark_model = None
mean_arr = None
std_arr = None
inv_labels = None

if LANDMARK_MODEL_PATH.exists() and SCALER_PATH.exists():
    try:
        model_state, mean_arr, std_arr, labels_map = load_model_and_scaler(str(LANDMARK_MODEL_PATH), str(SCALER_PATH), device)
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

# ------------------ Optional TorchScript model loading ------------------
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

# ------------------ Mediapipe Hands (static image mode) ------------------
mp_h = mp.solutions.hands
# Use static_image_mode=True because we process independent frames via HTTP -> avoids timestamp mismatch.
_HANDS_KW = dict(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_h.Hands(**_HANDS_KW)

# lock to avoid concurrent hands.process() calls (mediapipe solution graph not thread-safe)
_mp_lock = threading.Lock()

def _process_frame_safe(rgb_image):
    """
    Process RGB image with MediaPipe Hands in a thread-safe way.
    If a ValueError occurs specifically for timestamp/graph, recreate hands and retry once.
    """
    global hands
    with _mp_lock:
        try:
            return hands.process(rgb_image)
        except ValueError as e:
            msg = str(e)
            if "Packet timestamp mismatch" in msg or "Graph has errors" in msg:
                # recreate hands and retry once
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
    Expects JSON body: { "frame": "data:image/jpeg;base64,..." }
    Returns JSON: { "source": "...", "label": "...", "confidence": float }
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

    # Convert to RGB for Mediapipe
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Safe mediapipe processing
    try:
        results = _process_frame_safe(rgb)
    except Exception as e:
        # log and return server error
        print("MediaPipe processing error:", e)
        return jsonify({"error": "mediapipe processing error", "detail": str(e)}), 500

    # Extract 84-d landmarks (left+right flattened) using helper
    try:
        feat, left_p, right_p = extract_lr_landmarks(results)  # should return np.array(84,)
    except Exception as e:
        print("Feature extraction error:", e)
        return jsonify({"error": "feature extraction error", "detail": str(e)}), 500

    # Try word TS model first (if available) -- optional, may require adapting dims
    if word_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)  # shape (1, D)
            with torch.no_grad():
                out = word_ts(x)  # adapt depending on your export
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = word_labels[idx] if word_labels and idx < len(word_labels) else str(idx)
            return jsonify({"source": "word_ts", "label": label, "confidence": float(probs[idx] * 100)})
        except Exception:
            # fall through to next option
            pass

    # Try alphabet TorchScript if available
    if alpha_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = alpha_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            label = alpha_labels[idx] if alpha_labels and idx < len(alpha_labels) else (inv_labels.get(idx, str(idx)) if inv_labels else str(idx))
            return jsonify({"source": "alpha_ts", "label": label, "confidence": float(probs[idx] * 100)})
        except Exception:
            pass

    # Fallback: landmark MLP single-frame inference
    if landmark_model is None or mean_arr is None or std_arr is None:
        return jsonify({"error": "no landmark model available"}), 500

    try:
        x = (feat - mean_arr) / std_arr
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = landmark_model(x_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        idx = int(np.argmax(probs))
        label = inv_labels.get(idx, str(idx)) if inv_labels else str(idx)
        return jsonify({"source": "landmark_mlp", "label": label, "confidence": float(probs[idx] * 100)})
    except Exception as e:
        print("Model inference error:", e)
        return jsonify({"error": "model inference error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
