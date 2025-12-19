#!/usr/bin/env python3
# word/8_app_words_infer.py
"""
Real-time inference app for word model (fixed for MediaPipe Holistic outputs).

Usage:
  python .\word\8_app_words_infer.py --model models/word_lstm_aug/best.pt --camera 0 --device cuda --mirror --T 96 --run_every 4 --thresh 0.6
"""
import argparse, time, collections, numpy as np, torch, cv2, mediapipe as mp
from pathlib import Path

def extract_lr_pose_face(results):
    POSE_INDICES = [0,11,12,13,14,15,16,23,24]
    left = [0.0]*42; right=[0.0]*42; left_p=0; right_p=0
    if getattr(results, "left_hand_landmarks", None):
        left_p = 1
        left = [float(v) for lm in results.left_hand_landmarks.landmark for v in (lm.x, lm.y)]
    if getattr(results, "right_hand_landmarks", None):
        right_p = 1
        right = [float(v) for lm in results.right_hand_landmarks.landmark for v in (lm.x, lm.y)]
    pose = []
    if getattr(results, "pose_landmarks", None):
        for idx in POSE_INDICES:
            lm = results.pose_landmarks.landmark[idx]
            pose.extend([float(lm.x), float(lm.y)])
    else:
        pose = [0.0]*(len(POSE_INDICES)*2)
    # face bbox
    if getattr(results, "face_landmarks", None):
        xs = [float(lm.x) for lm in results.face_landmarks.landmark]
        ys = [float(lm.y) for lm in results.face_landmarks.landmark]
        x1,x2 = max(0.0,min(xs)), min(1.0,max(xs))
        y1,y2 = max(0.0,min(ys)), min(1.0,max(ys))
        face_box = [x1,y1,x2,y2]
    else:
        face_box = [0.0,0.0,0.0,0.0]
    return np.array(left,dtype=np.float32), np.array(right,dtype=np.float32), np.array(pose,dtype=np.float32), np.array(face_box,dtype=np.float32), np.array([left_p,right_p],dtype=np.int8)

def make_frame_feature(left, right, pose, present, lw_vel, rw_vel):
    dist = 0.0
    if left is not None and right is not None and left.size and right.size:
        lw = left[0:2].astype(np.float32)
        rw = right[0:2].astype(np.float32)
        try:
            dist = float(np.linalg.norm(lw - rw))
        except:
            dist = 0.0
    left_flat = left if left is not None else np.zeros(42,dtype=np.float32)
    right_flat = right if right is not None else np.zeros(42,dtype=np.float32)
    pose_flat = pose if pose is not None else np.zeros(18,dtype=np.float32)
    lw_vel = lw_vel if lw_vel is not None else np.zeros(2,dtype=np.float32)
    rw_vel = rw_vel if rw_vel is not None else np.zeros(2,dtype=np.float32)
    present = present if present is not None else np.array([0,0], dtype=np.float32)
    feat = np.concatenate([left_flat, right_flat, pose_flat, lw_vel.astype(np.float32), rw_vel.astype(np.float32), np.array([dist], dtype=np.float32), present.astype(np.float32)], axis=0)
    return feat

def compute_vel(prev_wrist, cur_wrist, fps):
    if prev_wrist is None or cur_wrist is None:
        return np.zeros(2, dtype=np.float32)
    dt = 1.0 / (fps if fps>0 else 30.0)
    return ((cur_wrist - prev_wrist) / dt).astype(np.float32)

def build_model_and_labels(model_path, device, D):
    ck = torch.load(model_path, map_location="cpu")
    labels = ck.get("labels", None) or ck.get("labels_map", None)
    if labels is None:
        raise RuntimeError("Model checkpoint missing labels")
    import torch.nn as nn
    class BiLSTM(nn.Module):
        def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, nclass))
        def forward(self,x,mask):
            out, _ = self.lstm(x)
            maskf = mask.unsqueeze(-1).float()
            summed = (out * maskf).sum(dim=1)
            denom = maskf.sum(dim=1).clamp(min=1e-3)
            pooled = summed / denom
            logits = self.head(pooled)
            return logits
    device_t = torch.device("cuda" if device=="cuda" and torch.cuda.is_available() else "cpu")
    model = BiLSTM(D, hidden=256, nlayers=2, nclass=len(labels), dropout=0.3).to(device_t)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, list(labels), device_t

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--run_every", type=int, default=4)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--min_active", type=int, default=6)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera", args.camera); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print("Camera fps:", fps)

    mp_hol = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    holistic = mp_hol.Holistic(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    feat_buffer = collections.deque(maxlen=args.T)
    mask_buffer = collections.deque(maxlen=args.T)
    prev_left_wrist = None
    prev_right_wrist = None

    sample_feat = np.zeros(109, dtype=np.float32)
    D = sample_feat.shape[0]
    model, labels, device = build_model_and_labels(args.model, args.device, D)

    frame_idx = 0
    last_pred = ("", 0.0, 0)

    print("Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if args.mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        left, right, pose, face_box, present = extract_lr_pose_face(results)
        cur_lw = left[0:2] if left is not None and left.size>1 else None
        cur_rw = right[0:2] if right is not None and right.size>1 else None
        lw_vel = compute_vel(prev_left_wrist, cur_lw, fps)
        rw_vel = compute_vel(prev_right_wrist, cur_rw, fps)
        prev_left_wrist = cur_lw
        prev_right_wrist = cur_rw

        feat = make_frame_feature(left, right, pose, present, lw_vel, rw_vel)
        feat_buffer.append(feat)
        mask_buffer.append(1 if (present is not None and present.sum()>0) else 0)

        # draw left/right hand landmarks if present
        try:
            if getattr(results, "left_hand_landmarks", None):
                mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if getattr(results, "right_hand_landmarks", None):
                mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if getattr(results, "pose_landmarks", None):
                mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_hol.POSE_CONNECTIONS)
        except Exception as e:
            # drawing errors shouldn't crash loop
            pass

        if frame_idx % args.run_every == 0 and len(feat_buffer) >= args.min_active:
            arr = np.stack(list(feat_buffer))
            L = arr.shape[0]
            if L < args.T:
                pad = np.zeros((args.T - L, D), dtype=np.float32)
                X = np.vstack([pad, arr])
                mask = np.array([0]*(args.T-L) + list(mask_buffer), dtype=np.uint8)
            else:
                X = arr[-args.T:]
                mask = np.array(list(mask_buffer)[-args.T:], dtype=np.uint8)
            xt = torch.from_numpy(X).unsqueeze(0).to(device)
            mt = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(xt, mt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            best = int(np.argmax(probs))
            prob = float(probs[best])
            label = labels[best]
            if prob >= args.thresh:
                last_pred = (label, prob, frame_idx)
            else:
                last_pred = ("", prob, frame_idx)

        display_text = f"{last_pred[0]} {last_pred[1]:.2f}" if last_pred[0] else "----"
        cv2.putText(frame, display_text, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if last_pred[0] else (0,0,255), 3)

        cv2.imshow("word_infer", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
