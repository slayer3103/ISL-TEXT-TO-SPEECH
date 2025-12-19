#!/usr/bin/env python3
# scripts/recorder_long_take.py
"""
Long-take recorder for repeated-sign capture.

Controls:
  s  : start recording (3s countdown)
  p  : pause / resume recording
  m  : mark current frame/time (writes to <video>.markers.csv)
  q  : stop and exit (video saved)

Usage:
  python .\word\1_recorder_long_take.py --signer p01 --label hello --out data/raw_recordings/p01 --width 1280 --height 720 --fps 30 --mirror
"""
import cv2, argparse, time, csv
from pathlib import Path
from datetime import datetime

def next_take_index(out_dir, signer, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{signer}_{label}_take*.mp4"))
    if not existing:
        return 1
    idxs = []
    for p in existing:
        name = p.stem
        parts = name.split("_")
        for part in parts:
            if part.startswith("take"):
                try:
                    idxs.append(int(part.replace("take","")))
                except:
                    pass
    return max(idxs)+1 if idxs else 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signer", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out", default="data/raw_recordings")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--codec", default="mp4v")
    p.add_argument("--mirror", action="store_true", help="mirror camera horizontally before saving/displaying")
    args = p.parse_args()

    signer = args.signer
    label = args.label
    base_out = Path(args.out) / signer
    base_out.mkdir(parents=True, exist_ok=True)

    take_idx = next_take_index(base_out, signer, label)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = f"{signer}_{label}_take{take_idx:02d}_{timestamp}.mp4"
    out_path = base_out / fname

    markers = []
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = None
    recording = False
    paused = False
    rec_frames = 0

    print("Controls: s=start, p=pause/resume, m=mark, q=quit/save")
    print("Press 's' to start recording (3s countdown).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame from camera")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        info = f"{'REC' if recording and not paused else 'PAUSED' if paused else 'READY'}  frames_rec={rec_frames}"
        cv2.putText(display, info, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if recording and not paused else (0,0,255), 2)
        cv2.putText(display, f"{signer} {label}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("recorder_long_take", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not recording:
            for i in (3,2,1):
                print(f"Starting in {i}...")
                time.sleep(1)
            writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (display.shape[1], display.shape[0]))
            recording = True
            paused = False
            rec_start = time.time()
            rec_frames = 0
            print("Recording started ->", out_path)
        elif key == ord('p') and recording:
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('m') and recording:
            t = time.time() - rec_start
            frame_no = rec_frames
            markers.append((frame_no, t))
            print(f"Marked frame {frame_no} (t={t:.3f}s)")
        elif key in (ord('q'), 27):
            print("Stopping.")
            break

        if recording and not paused:
            writer.write(frame)
            rec_frames += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    if markers:
        mark_path = out_path.with_suffix(".markers.csv")
        with open(mark_path, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame","time_s"])
            for fr, t in markers:
                wcsv.writerow([fr, f"{t:.4f}"])
        print("Wrote markers:", mark_path)
    print("Saved video:", out_path)
    print("Done.")

if __name__ == "__main__":
    main()
