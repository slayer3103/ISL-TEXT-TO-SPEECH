#!/usr/bin/env python3
# scripts/convert_npz_to_sequences.py
"""
Convert per-clip .npz (from video_to_clips) into fixed-length feature sequences (with presence flags and wrist velocities).

Usage:
  python .\word\4_convert_npz_to_sequences.py --labeled_dir data/word_sequences --out_dir data/word_sequences_ready --T 96

Output:
  data/word_sequences_ready/<label>/<orig_basename>_seq.npz
  each file contains:
    X : (T, D) float32
    mask : (T,) uint8  (1 for real frames, 0 for padded)
    meta : dict with original npz path, label, orig_len, fps
"""
import argparse, numpy as np, os
from pathlib import Path
from math import sqrt

def load_clip_npz(p):
    d = np.load(p, allow_pickle=True)
    left = d['hands_left']         # (F,42)
    right = d['hands_right']       # (F,42)
    pose = d['pose']               # (F,18)
    present = d['present']         # (F,2)
    ts = d['frame_ts']             # (F,)
    meta = d['meta'].item() if 'meta' in d.files else {}
    return left, right, pose, present, ts, meta

def compute_wrist_vel(left, right, fps):
    # left/right shape (F,42) -> wrist coords are first pair (index 0,1)
    if left.size == 0:
        F = right.shape[0]
    else:
        F = left.shape[0]
    lw = left.reshape(F,21,2)[:,0,:] if left.size else np.zeros((F,2),dtype=np.float32)
    rw = right.reshape(F,21,2)[:,0,:] if right.size else np.zeros((F,2),dtype=np.float32)
    # velocities = diff / dt ; pad first row with 0
    dt = 1.0 / (fps if fps>0 else 30.0)
    lw_dx = np.vstack([np.zeros((1,2),dtype=np.float32), (lw[1:]-lw[:-1]) / dt]) if F>1 else np.zeros((F,2),dtype=np.float32)
    rw_dx = np.vstack([np.zeros((1,2),dtype=np.float32), (rw[1:]-rw[:-1]) / dt]) if F>1 else np.zeros((F,2),dtype=np.float32)
    return lw_dx.astype(np.float32), rw_dx.astype(np.float32)

def hand_distance(left, right):
    # compute Euclidean distance between wrist points per frame
    if left.size == 0 or right.size == 0:
        return np.zeros((max(left.shape[0], right.shape[0]),), dtype=np.float32)
    F = left.shape[0]
    lw = left.reshape(F,21,2)[:,0,:]
    rw = right.reshape(F,21,2)[:,0,:]
    d = np.linalg.norm(lw - rw, axis=1).astype(np.float32)
    return d

def build_per_frame_features(left, right, pose, lw_vel, rw_vel, dist, present):
    # concatenation order:
    # left (42), right (42), pose (18), lw_vel (2), rw_vel (2), dist (1), present(2) => total 109
    F = left.shape[0]
    dist = dist.reshape(F,1)
    features = np.concatenate([left, right, pose, lw_vel, rw_vel, dist, present.astype(np.float32)], axis=1)
    return features.astype(np.float32)

def pad_or_trim(X, T):
    F, D = X.shape
    if F >= T:
        X2 = X[:T,:]
        mask = np.ones((T,), dtype=np.uint8)
    else:
        X2 = np.zeros((T, D), dtype=np.float32)
        X2[:F,:] = X
        mask = np.zeros((T,), dtype=np.uint8)
        mask[:F] = 1
    return X2, mask

def find_npz_files(labeled_dir):
    p = Path(labeled_dir)
    files = []
    for label_dir in sorted([d for d in p.iterdir() if d.is_dir()]):
        lbl = label_dir.name
        for npz in sorted(label_dir.glob("*.npz")):
            files.append((lbl, npz))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled_dir", required=True, help="dir with subfolders per label containing per-clip .npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--T", type=int, default=96, help="sequence length in frames")
    ap.add_argument("--fps_default", type=float, default=30.0)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_npz_files(args.labeled_dir)
    print(f"Found {len(files)} labeled .npz clips")
    for lbl, npz in files:
        try:
            left, right, pose, present, ts, meta = load_clip_npz(npz)
            fps = meta.get('fps', args.fps_default) if isinstance(meta, dict) else args.fps_default
            # ensure shapes: (F,42)/(F,18)/(F,2)
            F = max(left.shape[0], right.shape[0], pose.shape[0], present.shape[0], ts.shape[0] if ts is not None else 0)
            # pad per-clip arrays to same length F if needed
            def ensure_len(arr, expect):
                if arr.shape[0] == expect:
                    return arr
                D = arr.shape[1] if arr.ndim>1 else 1
                out = np.zeros((expect, D), dtype=arr.dtype)
                out[:arr.shape[0], ...] = arr
                return out
            left = ensure_len(left, F) if left.size else np.zeros((F,42),dtype=np.float32)
            right = ensure_len(right, F) if right.size else np.zeros((F,42),dtype=np.float32)
            pose = ensure_len(pose, F) if pose.size else np.zeros((F,18),dtype=np.float32)
            present = ensure_len(present, F) if present.size else np.zeros((F,2),dtype=np.int32)
            ts = ensure_len(ts.reshape(-1,1), F).flatten() if ts is not None and ts.size else np.arange(F)/fps

            lw_vel, rw_vel = compute_wrist_vel(left, right, fps)
            dist = hand_distance(left, right)
            X = build_per_frame_features(left, right, pose, lw_vel, rw_vel, dist, present)
            X2, mask = pad_or_trim(X, args.T)

            out_dir_label = out_root / lbl
            out_dir_label.mkdir(parents=True, exist_ok=True)
            out_name = npz.stem + "_seq.npz"
            out_path = out_dir_label / out_name
            meta_out = {"orig_npz": str(npz), "label": lbl, "orig_len": int(F), "fps": float(fps)}
            np.savez_compressed(str(out_path), X=X2, mask=mask, meta=meta_out)
            print("Wrote:", out_path, "label:", lbl, "frames:", F, "->", args.T)
        except Exception as e:
            print("ERROR processing", npz, ":", e)

    print("Done converting all clips.")

if __name__ == "__main__":
    main()
