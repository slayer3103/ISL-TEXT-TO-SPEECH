#!/usr/bin/env python3
"""
backend_api_multi.py - Flask backend that supports:
 - single-frame inference (alphabet/landmark MLP / TorchScript models)
 - sequence (multi-frame) inference for word LSTM (endpoint /infer_seq)
 - audio caching (pyttsx3)
 - mediapipe thread-safe processing (streaming mode for sequence)
 - serves frontend from static/frontend

Debug:
 - set DEBUG_FEATURES=1 to enable file debug dumps under static/debug_logs/
 - the server also keeps the last debug payload in memory and exposes it at GET /debug_last

 $env:DEBUG_FEATURES = "1"; python backend.py
"""

import os
import base64
import json
import threading
import hashlib
import time
from pathlib import Path
from typing import Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import numpy as np
import cv2
import torch
import mediapipe as mp
import joblib

# -------------------------
# Helpers fallback if app_isl not found
# -------------------------
try:
    from app_isl import build_model as build_landmark_mlp, extract_lr_landmarks, load_model_and_scaler
except Exception:
    def extract_lr_landmarks(results):
        left = [0.0] * 42
        right = [0.0] * 42
        left_p = 0
        right_p = 0
        if getattr(results, "multi_hand_landmarks", None) and getattr(results, "multi_handedness", None):
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

# -------------------------
# Paths + Flask init
# -------------------------
ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "models"
LANDMARK_MODEL_DIR = MODEL_DIR / "landmark_model"

AUDIO_CACHE_DIR = ROOT / "static" / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = ROOT / "static" / "frontend"
DEBUG_LOG_DIR = ROOT / "static" / "debug_logs"
DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(ROOT / "static"))
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Backend device:", device)

# -------------------------
# Config and debug flags
# -------------------------
PROB_THRESH = float(os.environ.get("PROB_THRESH", 0.6))
USE_GUARD = True
DEBUG = os.environ.get("DEBUG_FEATURES", "0") == "1"
print("DEBUG_FEATURES:", DEBUG)

# In-memory debug payloads (always set when frames were processed)
LAST_DEBUG_PAYLOAD = None
LAST_DEBUG_TS = None

# -------------------------
# Load landmark model + scaler (optional)
# -------------------------
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

# -------------------------
# TorchScript optional models (alphabet + word)
# -------------------------
WORD_TS_PATH = MODEL_DIR / "word_model" / "export" / "model_ts.pt"
ALPHA_TS_PATH = MODEL_DIR / "alphabet_model" / "export" / "model_ts.pt"

alpha_ts = None; alpha_labels = None
word_ts = None; word_ts_labels = None

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

# -------------------------
# LSTM checkpoint (word model) optional
# -------------------------
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

# -------------------------
# Mediapipe setup
# -------------------------
mp_hol = mp.solutions.holistic
mp_h = mp.solutions.hands
_HANDS_KW = dict(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Holistic in streaming mode for sequences:
_holistic_kw_stream = dict(static_image_mode=False, model_complexity=1,
                           refine_face_landmarks=False,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)

hands = mp_h.Hands(**_HANDS_KW)
hol = mp_hol.Holistic(**_holistic_kw_stream)
_mp_lock = threading.Lock()

def _process_frame_safe_hands(rgb_image):
    global hands
    with _mp_lock:
        try:
            rgb_image.flags.writeable = False
            res = hands.process(rgb_image)
            rgb_image.flags.writeable = True
            return res
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
            rgb_image.flags.writeable = False
            res = hol.process(rgb_image)
            rgb_image.flags.writeable = True
            return res
        except ValueError as e:
            msg = str(e)
            if "Packet timestamp mismatch" in msg or "Graph has errors" in msg:
                try:
                    hol.close()
                except Exception:
                    pass
                hol = mp_hol.Holistic(static_image_mode=False,
                                      model_complexity=_holistic_kw_stream.get("model_complexity", 1),
                                      refine_face_landmarks=False,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
                rgb_image.flags.writeable = False
                res = hol.process(rgb_image)
                rgb_image.flags.writeable = True
                return res
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

# -------------------------
# audio utils (pyttsx3 caching)
# -------------------------
def _text_to_hash_filename(text: str):
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"{h}.wav"

def _synthesize_wav_pyttsx3(text: str, out_path: Path):
    try:
        import pyttsx3
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

# -------------------------
# sequence feature helpers (layout = 109)
# -------------------------
POSE_IDX = [0,11,12,13,14,15,16,23,24]

def extract_from_holistic(results) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

# -------------------------
# API endpoints
# -------------------------
@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "device": str(device),
        "has_landmark_model": bool(landmark_model is not None),
        "has_alpha_ts": bool(alpha_ts is not None),
        "has_word_ts": bool(word_ts is not None),
        "has_word_lstm": bool(word_lstm_model is not None),
        "debug": DEBUG
    })

@app.route("/")
def index():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return send_from_directory(str(FRONTEND_DIR), "index.html")
    return jsonify({"ok": True, "note": "no frontend found in static/frontend. Place your files there."})

@app.route("/infer", methods=["POST"])
def infer():
    """
    Single-frame inference endpoint.
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
            resp["audio_url"] = url; resp["audio_ready"] = exists
        return jsonify(resp)

    # Try word TorchScript first (if available)
    if word_ts is not None:
        try:
            x = torch.from_numpy(feat.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                out = word_ts(x)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs)); prob = float(probs[idx])
            label = word_ts_labels[idx] if word_ts_labels and idx < len(word_ts_labels) else str(idx)
            out_label = label if prob >= thresh else "<UNKNOWN>"
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
            out_label = label if prob >= thresh else "<UNKNOWN>"
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
        out_label = label if prob >= thresh else "<UNKNOWN>"
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
    Input JSON: {"frames": ["data:image/jpeg;base64,...", ...], "speak": bool, "thresh": 0.6}
    """
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({"error": "invalid json", "detail": str(e)}), 400

    frames = data.get("frames")
    if not frames or not isinstance(frames, list):
        return jsonify({"error": "frames must be a non-empty list of base64 images"}), 400

    thresh = float(data.get("thresh", PROB_THRESH))

    # Quick runtime trace
    print(f"[infer_seq] frames_received={len(frames)} thresh={thresh} debug={DEBUG}")

    # ---- tunables ----
    TARGET_RES = (320, 240)  # resize for consistent scaling and speed
    fps_approx = float(os.environ.get("SEQ_FPS_APPROX", 30.0))
    # ------------------

    # First try LSTM checkpoint model if loaded
    if word_lstm_model is not None:
        feats = []
        presents = []
        prev_l_full = None
        prev_r_full = None
        frame_idx = 0

        for b64 in frames:
            frame_idx += 1
            img = _decode_b64_frame(b64)
            if img is None:
                print(f"[infer_seq] skipped frame {frame_idx}: decode failed")
                continue

            # Resize for consistent landmark scale
            try:
                img = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                res = _process_frame_safe_holistic(rgb)
            except Exception as e:
                print("holistic processing error:", e)
                continue

            left, right, pose, present = extract_from_holistic(res)

            # velocities (first two coords) scaled by fps_approx
            lw_vel = np.zeros(2, np.float32)
            rw_vel = np.zeros(2, np.float32)
            if prev_l_full is not None and left.size:
                lw_vel = (left[:2] - prev_l_full[:2]) * fps_approx
            if prev_r_full is not None and right.size:
                rw_vel = (right[:2] - prev_r_full[:2]) * fps_approx

            prev_l_full = left if left.size else prev_l_full
            prev_r_full = right if right.size else prev_r_full

            feat = make_feat(left, right, pose, present, lw_vel, rw_vel)
            feats.append(feat.tolist())
            presents.append(1 if present.sum() > 0 else 0)

        if len(feats) == 0:
            print("[infer_seq] no valid frames processed -> returning 400")
            return jsonify({"error": "no valid frames processed"}), 400

        # Convert to numpy for model
        X = np.stack([np.array(f, dtype=np.float32) for f in feats], axis=0)  # (T,D)
        M = np.array(presents, dtype=np.uint8)

        # Prepare in-memory debug payload (always populate when processed frames exist)
        try:
            meta_payload = {"frames_received": len(frames), "processed_frames": int(X.shape[0]),
                            "TARGET_RES": TARGET_RES, "fps_approx": fps_approx}
            payload = {"feats_arr": X.tolist(), "mask": M.tolist(), "meta": meta_payload}
            global LAST_DEBUG_PAYLOAD, LAST_DEBUG_TS
            LAST_DEBUG_PAYLOAD = payload
            LAST_DEBUG_TS = time.strftime("%Y-%m-%dT%H-%M-%S")
            print(f"[infer_seq] stored LAST_DEBUG_TS={LAST_DEBUG_TS} processed_frames={X.shape[0]}")
        except Exception as e:
            print("Failed preparing in-memory debug payload:", e)

        # File debug dump if requested
        if DEBUG:
            try:
                ts = time.strftime("%Y-%m-%dT%H-%M-%S")
                fname = DEBUG_LOG_DIR / f"log_backend_{ts}.json"
                with open(fname, "w", encoding="utf-8") as fo:
                    json.dump(payload, fo)
                print("Wrote debug log to:", str(fname))
            except Exception as e:
                print("Failed writing debug log:", e)

        # Model inference
        xt = torch.from_numpy(X).unsqueeze(0).to(word_lstm_device)
        mt = torch.from_numpy(M).unsqueeze(0).to(word_lstm_device)
        try:
            with torch.inference_mode():
                logits = word_lstm_model(xt, mt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax()); prob = float(probs[idx])
            label = word_lstm_labels[idx] if word_lstm_labels and idx < len(word_lstm_labels) else str(idx)
            out_label = label if prob >= thresh else "<UNKNOWN>"
            resp = {"source": "word_lstm", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_lstm inference error:", e)
            # Fallthrough to try torchscript fallback

    # Fallback: if word TorchScript exists, try single-frame inference on last frame
    if word_ts is not None:
        last = frames[-1]
        img = _decode_b64_frame(last)
        if img is None:
            return jsonify({"error": "could not decode last frame for fallback"}), 400
        try:
            img = cv2.resize(img, TARGET_RES, interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass
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
            out_label = label if prob >= thresh else "<UNKNOWN>"
            resp = {"source": "word_ts_fallback", "label": out_label, "confidence": float(prob * 100)}
            if data.get("speak", False):
                exists, url = _ensure_audio_cached(out_label)
                resp["audio_url"] = url; resp["audio_ready"] = exists
            return jsonify(resp)
        except Exception as e:
            print("word_ts fallback error:", e)

    return jsonify({"error": "no sequence model available"}), 500

@app.route("/debug_last", methods=["GET"])
def debug_last():
    """
    Return the last in-memory debug payload (feats_arr, mask, meta).
    Useful when file writes are not observable / permission blocked.
    """
    global LAST_DEBUG_PAYLOAD, LAST_DEBUG_TS
    if LAST_DEBUG_PAYLOAD is None:
        return jsonify({"ok": False, "msg": "no debug payload captured yet"}), 404
    return jsonify({"ok": True, "ts": LAST_DEBUG_TS, "payload": LAST_DEBUG_PAYLOAD})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
