
"""
Assisted short-clip recorder (operator presses 'c' per repetition or 'r' for variable recording).

Controls:
  c : capture a fixed-duration clip (default clip_dur seconds)
  r : toggle start/stop variable-length recording
  n : change label (interactive)
  q : quit

Usage:
  python .\word\1_recorder_clips_assisted.py --signer p01 --label hello --out data/raw_recordings/p01 --clip_dur 1.2 --width 1280 --height 720 --fps 30 --mirror
"""
import cv2, argparse, time
from pathlib import Path
from datetime import datetime

def next_clip_index(out_dir, signer, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{signer}_{label}_clip*.mp4"))
    if not existing:
        return 1
    idxs = []
    for p in existing:
        name = p.stem
        parts = name.split("_")
        for part in parts:
            if part.startswith("clip"):
                try:
                    idxs.append(int(part.replace("clip","")))
                except:
                    pass
    return max(idxs)+1 if idxs else 1

def format_fname(signer,label,kind,idx):
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{signer}_{label}_{kind}{idx:04d}_{ts}.mp4"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signer", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="data/raw_recordings")
    ap.add_argument("--clip_dur", type=float, default=1.2)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--codec", default="mp4v")
    ap.add_argument("--mirror", action="store_true", help="mirror camera horizontally before saving/displaying")
    args = ap.parse_args()

    signer = args.signer
    label = args.label
    base_out = Path(args.out)
    base_out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    running = True
    rec_var = False
    var_writer = None
    var_start_time = None
    var_frames = 0

    print("Assisted clip recorder:")
    print("  c: capture fixed clip (duration {:.2f}s)".format(args.clip_dur))
    print("  r: toggle start/stop variable-length recording")
    print("  n: change label")
    print("  q: quit")

    fixed_idx = next_clip_index(base_out, signer, label)
    var_idx = 1

    while running:
        ret, frame = cap.read()
        if not ret:
            print("No frame from camera")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        status = "REC_VAR" if rec_var else "READY"
        cv2.putText(display, f"{status} label={label} fixed_idx={fixed_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if not rec_var else (0,0,255), 2)
        cv2.imshow("recorder_clips_assisted", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            clip_frames = []
            frames_to_capture = int(max(1, round(args.clip_dur * args.fps)))
            print(f"Capturing fixed clip for {args.clip_dur}s -> {frames_to_capture} frames")
            for f in range(frames_to_capture):
                ret2, fr = cap.read()
                if not ret2: break
                if args.mirror:
                    fr = cv2.flip(fr, 1)
                clip_frames.append(fr.copy())
                time.sleep(max(0, (1.0/args.fps) - 0.001))
            fname = format_fname(signer, label, "clip", fixed_idx)
            outp = base_out / fname
            writer = cv2.VideoWriter(str(outp), fourcc, args.fps, (args.width, args.height))
            for fr in clip_frames:
                writer.write(fr)
            writer.release()
            print("Saved fixed clip:", outp)
            fixed_idx += 1

        elif key == ord('r'):
            if not rec_var:
                fname = format_fname(signer, label, "var", var_idx)
                outp = base_out / fname
                var_writer = cv2.VideoWriter(str(outp), fourcc, args.fps, (args.width, args.height))
                rec_var = True
                var_start_time = time.time()
                var_frames = 0
                var_path = outp
                print("Started variable recording ->", outp)
            else:
                rec_var = False
                var_writer.release()
                elapsed = time.time() - var_start_time
                print(f"Stopped variable recording. frames={var_frames} time={elapsed:.2f}s saved to {var_path}")
                var_idx += 1

        elif key == ord('n'):
            new = input("New label: ").strip()
            if new:
                label = new
                fixed_idx = next_clip_index(base_out, signer, label)
                print("Label changed to", label)

        elif key in (ord('q'), 27):
            print("Quitting.")
            running = False
            break

        if rec_var and var_writer is not None:
            write_frame = frame if not args.mirror else cv2.flip(frame,1)
            var_writer.write(write_frame)
            var_frames += 1

    cap.release()
    if var_writer:
        var_writer.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
