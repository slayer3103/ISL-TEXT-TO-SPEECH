#!/usr/bin/env python3
# 8_app_words_infer_fast.py (updated) - debug/log dump removed
# Example:
# python ./word/8_app_words_infer_fast2.py --model models/word_lstm_aug/best.pt --camera 0 --device cuda --mirror --width 640 --height 360 --T 96 --run_every 3 --thresh 0.6 --smooth_tau 0.6

import argparse, time, collections, threading, json, os
import numpy as np, torch, cv2, mediapipe as mp

# ---------------------------
# feature helpers (same layout as backend)
# ---------------------------
POSE_IDX = [0,11,12,13,14,15,16,23,24]

def extract_lr_pose(results):
    left = np.zeros(42, dtype=np.float32); right = np.zeros(42, dtype=np.float32)
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

def make_feat(left,right,pose,present,lw_vel,rw_vel):
    dist = 0.0
    if left.size and right.size:
        dist = float(np.linalg.norm(left[:2]-right[:2]))
    return np.concatenate([
        left if left.size else np.zeros(42,np.float32),
        right if right.size else np.zeros(42,np.float32),
        pose if pose.size else np.zeros(18,np.float32),
        lw_vel.astype(np.float32),
        rw_vel.astype(np.float32),
        np.array([dist],np.float32),
        present.astype(np.float32)
    ], axis=0)

# ---------------------------
# velocity helper using measured dt
# ---------------------------
def compute_velocity(prev_xy, cur_xy, dt, fallback_fps=30.0):
    if prev_xy is None or cur_xy is None or not cur_xy.any():
        return np.zeros(2, dtype=np.float32)
    if dt is None or dt <= 0:
        fps = fallback_fps
    else:
        fps = 1.0/dt
    return ((cur_xy - prev_xy) * fps).astype(np.float32) if prev_xy is not None else np.zeros(2, dtype=np.float32)

# ---------------------------
# camera thread (unchanged)
# ---------------------------
class CameraThread:
    def __init__(self, cam_idx, width, height, backend, mirror):
        self.cap = cv2.VideoCapture(cam_idx, backend)
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.mirror = mirror
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.t = threading.Thread(target=self.loop, daemon=True)
    def loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            if self.mirror: f = cv2.flip(f,1)
            with self.lock:
                self.frame = f
    def start(self):
        self.t.start()
    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()
    def release(self):
        self.stopped = True
        try:
            self.t.join(timeout=0.2)
        except:
            pass
        self.cap.release()

# ---------------------------
# model builder (same as earlier)
# ---------------------------
def build_model(model_path, device, D, nclass=None):
    ck = torch.load(model_path, map_location="cpu")
    labels = ck.get("labels") or ck.get("labels_map")
    if labels is None:
        # if checkpoint doesn't provide labels, expect external usage
        raise RuntimeError("Checkpoint missing labels")
    if nclass is None: nclass = len(labels)
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
    dev = torch.device("cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    model = BiLSTM(D, nclass=nclass).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, list(labels), dev

# ---------------------------
# non-blocking TTS helper using pyttsx3
# ---------------------------
class TTSSpeaker:
    def __init__(self, cooldown=1.2):
        self.cooldown = cooldown
        self.last_spoken = 0.0
        self.lock = threading.Lock()
    def speak(self, text):
        now = time.time()
        if now - self.last_spoken < self.cooldown:
            return
        self.last_spoken = now
        t = threading.Thread(target=self._job, args=(text,), daemon=True)
        t.start()
    def _job(self, text):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(str(text))
            engine.runAndWait()
        except Exception as e:
            # silent on TTS error (no logging)
            pass

# ---------------------------
# main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--backend", type=str, default="dshow", choices=["dshow","msmf","any"])
    ap.add_argument("--T", type=int, default=96)
    ap.add_argument("--run_every", type=int, default=3)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--no_draw", action="store_true")
    ap.add_argument("--smooth_tau", type=float, default=0.5, help="EMA smoothing for probs [0..1], 0=no smoothing")
    ap.add_argument("--no_tts", action="store_true", help="disable speech")
    ap.add_argument("--target_res", type=int, nargs=2, metavar=('W','H'), default=None, help="Optional internal processing resize (W H)")
    ap.add_argument("--fps_approx", type=float, default=30.0, help="fallback fps for velocity computation")
    args = ap.parse_args()

    backend = cv2.CAP_DSHOW if args.backend=="dshow" else (cv2.CAP_MSMF if args.backend=="msmf" else 0)
    cam = CameraThread(args.camera, args.width, args.height, backend, args.mirror); cam.start()
    time.sleep(0.05)  # warmup

    mp_hol = mp.solutions.holistic
    mp_hands = mp.solutions.hands
    hol = mp_hol.Holistic(static_image_mode=False, model_complexity=0,
                          refine_face_landmarks=False,
                          min_detection_confidence=0.5, min_tracking_confidence=0.5)

    D = 109
    buf_feats = collections.deque(maxlen=args.T)
    buf_mask  = collections.deque(maxlen=args.T)
    prev_lw = None; prev_rw = None
    prev_time = None

    model, labels, dev = build_model(args.model, args.device, D)
    last_ema = None
    last_label = ""
    last_prob  = 0.0

    # tts
    tts = None if args.no_tts else TTSSpeaker(cooldown=1.2)

    print("Press ESC to quit.")
    while True:
        frame = cam.read()
        if frame is None:
            cv2.waitKey(1); continue

        if args.target_res:
            try:
                frame = cv2.resize(frame, tuple(args.target_res), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.time()
        dt = None if prev_time is None else (now - prev_time)
        prev_time = now

        res = hol.process(rgb)

        left,right,pose,present = extract_lr_pose(res)
        lw = left[:2] if left.size else None
        rw = right[:2] if right.size else None

        lw_v = compute_velocity(prev_lw, lw, dt, fallback_fps=args.fps_approx) if lw is not None else np.zeros(2,np.float32)
        rw_v = compute_velocity(prev_rw, rw, dt, fallback_fps=args.fps_approx) if rw is not None else np.zeros(2,np.float32)

        prev_lw = lw; prev_rw = rw

        feat = make_feat(left,right,pose,present,lw_v,rw_v)
        buf_feats.append(feat)
        buf_mask.append(1 if present.sum()>0 else 0)

        if not args.no_draw:
            try:
                if getattr(res,"left_hand_landmarks",None):
                    mp.solutions.drawing_utils.draw_landmarks(frame,res.left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if getattr(res,"right_hand_landmarks",None):
                    mp.solutions.drawing_utils.draw_landmarks(frame,res.right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if getattr(res,"pose_landmarks",None):
                    mp.solutions.drawing_utils.draw_landmarks(frame,res.pose_landmarks, mp_hol.POSE_CONNECTIONS)
            except Exception:
                pass

        if (len(buf_feats)==args.T) and ((len(buf_feats) % args.run_every)==0):
            X = np.stack(buf_feats, axis=0)   # (T,D)
            M = np.array(buf_mask, dtype=np.uint8)  # (T,)
            xt = torch.from_numpy(X).unsqueeze(0).to(dev)
            mt = torch.from_numpy(M).unsqueeze(0).to(dev)

            with torch.inference_mode():
                if dev.type=="cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        logits = model(xt, mt)
                else:
                    logits = model(xt, mt)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # smoothing (EMA)
            if args.smooth_tau>0:
                if last_ema is None: last_ema = probs.copy()
                else: last_ema = args.smooth_tau*probs + (1-args.smooth_tau)*last_ema
                use = last_ema
            else:
                use = probs

            idx = int(use.argmax())
            prob = float(use[idx])
            lab  = labels[idx]
            if prob >= args.thresh:
                last_label, last_prob = lab, prob
                if tts is not None:
                    tts.speak(last_label)
            else:
                last_label, last_prob = "", prob

        txt = f"{last_label} {last_prob:.2f}" if last_label else "----"
        cv2.putText(frame, txt, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0) if last_label else (0,0,255), 3)
        cv2.imshow("word_infer_fast", frame)
        if (cv2.waitKey(1) & 0xFF) == 27: break

    cam.release()
    hol.close()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
