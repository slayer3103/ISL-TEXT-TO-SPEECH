# scripts/record_word_sequences.py
"""
Interactive recorder for word-sequence data using MediaPipe Holistic.

Saves each clip (start..stop) as a .npz containing:
  - hands_left: (T,42)
  - hands_right: (T,42)
  - pose: (T,18)
  - face_bbox: (T,4)
  - present: (T,2) (left/right present flags)
  - frame_ts: (T,)
  - annotations: list of dicts {frame_idx, label, time}
  - meta: dict {fps, signer, camera_idx, start_time, end_time, frames}

Keys while running:
  SPACE : start/stop recording the current clip
  m     : mark boundary in current clip (prompts for optional label)
  v     : toggle saving raw mp4 for each saved clip
  q/ESC : quit program
"""

#  python .\scripts\record_word_sequences.py --signer p01 --camera_idx 0


import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from pathlib import Path

# which pose indices to keep (MediaPipe Pose 33). These give shoulders/elbows/wrists/hips/nose etc.
POSE_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # 9 points -> 18 floats

OUT_DIR = Path("data/word_clips")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_selected_pose(pose_landmarks):
    pts = []
    if pose_landmarks is None:
        return [0.0] * (len(POSE_INDICES) * 2)
    for idx in POSE_INDICES:
        lm = pose_landmarks.landmark[idx]
        pts.extend([float(lm.x), float(lm.y)])
    return pts

def face_bbox_from_landmarks(face_landmarks):
    if face_landmarks is None:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [float(lm.x) for lm in face_landmarks.landmark]
    ys = [float(lm.y) for lm in face_landmarks.landmark]
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    x1 = max(0.0, min(xs)); x2 = min(1.0, max(xs))
    y1 = max(0.0, min(ys)); y2 = min(1.0, max(ys))
    return [x1, y1, x2, y2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--signer", type=str, default="p01")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--min_det", type=float, default=0.5)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(static_image_mode=False,
                                    model_complexity=1,
                                    min_detection_confidence=args.min_det,
                                    min_tracking_confidence=0.5)

    recording = False
    save_video_flag = False
    out_video_writer = None
    clip_idx = 0

    # buffers per clip
    clip_left = []
    clip_right = []
    clip_pose = []
    clip_face = []
    clip_present = []
    clip_ts = []
    annotations = []
    start_time = None

    print("Ready. SPACE to start/stop a clip. 'm' to mark boundary. 'v' toggle saving mp4 clips. 'q' to quit.")
    frame_count_total = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed or closed.")
            break
        frame_count_total += 1
        image = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # extract hands left/right flattened (42 values each)
        left_hand = [0.0]*42
        right_hand = [0.0]*42
        left_p = 0
        right_p = 0
        if results.left_hand_landmarks:
            left_p = 1
            pts = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
            flat = [float(x) for pair in pts for x in pair]
            if len(flat) == 42:
                left_hand = flat
        if results.right_hand_landmarks:
            right_p = 1
            pts = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
            flat = [float(x) for pair in pts for x in pair]
            if len(flat) == 42:
                right_hand = flat

        pose_sel = extract_selected_pose(results.pose_landmarks)
        face_box = face_bbox_from_landmarks(results.face_landmarks)

        # draw landmarks for visual feedback
        debug = image.copy()
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(debug, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(debug, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(debug, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
        if results.face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(debug, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)

        # overlay status text
        if recording:
            cv2.putText(debug, f"REC {args.signer} clip#{clip_idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.putText(debug, "Press 'm' to mark boundary | 'v' toggle mp4 | 'space' to stop", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        else:
            cv2.putText(debug, f"Idle - Press SPACE to start recording (signer: {args.signer})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Recorder", debug)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        if key == 32:  # SPACE toggles recording
            if not recording:
                # start recording
                recording = True
                clip_left = []
                clip_right = []
                clip_pose = []
                clip_face = []
                clip_present = []
                clip_ts = []
                annotations = []
                start_time = time.time()
                clip_idx += 1
                if save_video_flag:
                    video_fname = OUT_DIR / f"{args.signer}_clip{clip_idx:04d}_{int(start_time)}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out_video_writer = cv2.VideoWriter(str(video_fname), fourcc, fps, (frame.shape[1], frame.shape[0]))
                print(f"Started recording clip #{clip_idx}")
            else:
                # stop & save
                recording = False
                end_time = time.time()
                hands_left_arr = np.stack(clip_left) if clip_left else np.zeros((0,42),dtype=np.float32)
                hands_right_arr = np.stack(clip_right) if clip_right else np.zeros((0,42),dtype=np.float32)
                pose_arr = np.stack(clip_pose) if clip_pose else np.zeros((0,len(POSE_INDICES)*2), dtype=np.float32)
                face_arr = np.stack(clip_face) if clip_face else np.zeros((0,4), dtype=np.float32)
                present_arr = np.stack(clip_present) if clip_present else np.zeros((0,2), dtype=np.int32)
                ts_arr = np.array(clip_ts, dtype=np.float32)
                base = f"{args.signer}_clip{clip_idx:04d}_{int(start_time)}"
                npz_path = OUT_DIR / f"{base}.npz"
                meta = {"fps": fps, "signer": args.signer, "camera_idx": args.camera_idx, "start_time": start_time, "end_time": end_time, "frames": int(len(ts_arr))}
                np.savez_compressed(str(npz_path),
                                    hands_left=hands_left_arr,
                                    hands_right=hands_right_arr,
                                    pose=pose_arr,
                                    face_bbox=face_arr,
                                    present=present_arr,
                                    frame_ts=ts_arr,
                                    annotations=annotations,
                                    meta=meta)
                if save_video_flag and out_video_writer is not None:
                    out_video_writer.release()
                    out_video_writer = None
                    print(f"Saved video for clip: {video_fname}")
                print(f"Saved clip: {npz_path} frames={len(ts_arr)} annotations={len(annotations)}")
        elif key == ord('m') and recording:
            cur_idx = len(clip_ts)
            label = input("Boundary label (or Enter to leave blank): ").strip()
            annotations.append({"frame_idx": cur_idx, "label": label if label else None, "time": time.time()})
            print("Marked boundary:", annotations[-1])
        elif key == ord('v'):
            save_video_flag = not save_video_flag
            print("Save mp4 for each clip:", save_video_flag)

        # append per-frame data if recording
        if recording:
            clip_left.append(np.array(left_hand, dtype=np.float32))
            clip_right.append(np.array(right_hand, dtype=np.float32))
            clip_pose.append(np.array(pose_sel, dtype=np.float32))
            clip_face.append(np.array(face_box, dtype=np.float32))
            clip_present.append(np.array([left_p, right_p], dtype=np.int32))
            clip_ts.append(time.time())
            if save_video_flag and out_video_writer is not None:
                out_video_writer.write(frame)

    cap.release()
    holistic.close()
    cv2.destroyAllWindows()
    print("Recorder exited. Total frames seen:", frame_count_total)

if __name__ == "__main__":
    main()
