#!/usr/bin/env python3
# scripts/batch_extract_npz.py
"""
Batch extract MediaPipe landmark npz files from a folder of mp4 clips.

Usage:
  python .\word\2_batch_extract_npz.py --in_dir data/raw_recordings/p01 --out_dir data/clip_npz --mirror

Outputs:
  <out_dir>/<origname>.npz  with arrays:
    hands_left (F,42), hands_right (F,42), pose (F,18), face_bbox (F,4), present (F,2), frame_ts (F,), meta (dict)
"""
import argparse, cv2, numpy as np, os
from pathlib import Path
import mediapipe as mp

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
    if getattr(results, "face_landmarks", None):
        xs = [float(lm.x) for lm in results.face_landmarks.landmark]
        ys = [float(lm.y) for lm in results.face_landmarks.landmark]
        x1,x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1,y2 = max(0.0, min(ys)), min(1.0, max(ys))
        face_box = [x1,y1,x2,y2]
    else:
        face_box = [0.0,0.0,0.0,0.0]
    return np.array(left, dtype=np.float32), np.array(right, dtype=np.float32), np.array(pose, dtype=np.float32), np.array(face_box, dtype=np.float32), np.array([left_p,right_p], dtype=np.int8)

def process_file(in_path, out_path, mirror=False, model_complex=1, min_det=0.5):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print("ERROR: cannot open", in_path); return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    mp_hol = mp.solutions.holistic
    hol = mp_hol.Holistic(static_image_mode=False, model_complexity=model_complex,
                          min_detection_confidence=min_det, min_tracking_confidence=0.5)
    left_list=[]; right_list=[]; pose_list=[]; face_list=[]; present_list=[]; ts_list=[]
    frame_idx=0
    prev_time=0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hol.process(rgb)
        left, right, pose, face_box, present = extract_lr_pose_face(results)
        t = frame_idx / (fps or 30.0)
        left_list.append(left)
        right_list.append(right)
        pose_list.append(pose)
        face_list.append(face_box)
        present_list.append(present)
        ts_list.append(t)
    cap.release()
    hol.close()
    # stack & save
    left_arr = np.stack(left_list) if left_list else np.zeros((0,42),dtype=np.float32)
    right_arr = np.stack(right_list) if right_list else np.zeros((0,42),dtype=np.float32)
    pose_arr = np.stack(pose_list) if pose_list else np.zeros((0,18),dtype=np.float32)
    face_arr = np.stack(face_list) if face_list else np.zeros((0,4),dtype=np.float32)
    present_arr = np.stack(present_list) if present_list else np.zeros((0,2),dtype=np.int8)
    ts_arr = np.array(ts_list, dtype=np.float32)
    meta = {"video_source": str(in_path), "fps": float(fps)}
    np.savez_compressed(str(out_path),
                        hands_left=left_arr, hands_right=right_arr, pose=pose_arr,
                        face_bbox=face_arr, present=present_arr, frame_ts=ts_arr, meta=meta)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="folder with mp4 clips")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--min_det", type=float, default=0.5)
    ap.add_argument("--model_complex", type=int, default=1)
    args = ap.parse_args()

    in_root = Path(args.in_dir)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)
    mp4s = sorted([p for p in in_root.glob("*.mp4")])
    print("Found", len(mp4s), "mp4 files")
    for p in mp4s:
        outp = out_root / (p.stem + ".npz")
        if outp.exists():
            print("Skipping existing:", outp); continue
        print("Processing:", p)
        ok = process_file(p, outp, mirror=args.mirror, model_complex=args.model_complex, min_det=args.min_det)
        if ok:
            print("Saved:", outp)
        else:
            print("Failed:", p)

if __name__ == "__main__":
    main()
