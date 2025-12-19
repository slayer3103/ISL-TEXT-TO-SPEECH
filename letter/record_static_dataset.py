# python .\scripts\record_dataset.py

import cv2 as cv
import csv
import os
import time
import mediapipe as mp
from pathlib import Path

OUTPUT_CSV = "data/recorded_keypoints_test.csv"
SAVE_CROPS_DIR = "data/recorded_crops"  # optional; created if --save_crops
MAX_HANDS = 2

HEADER = None
for_side_header = []
# build coords header for 21 landmarks (x,y)
for i in range(21):
    for_side_header += [f"x{i}", f"y{i}"]

def ensure_out_dirs(save_crops):
    Path("data").mkdir(parents=True, exist_ok=True)
    if save_crops:
        Path(SAVE_CROPS_DIR).mkdir(parents=True, exist_ok=True)

def make_csv_header():
    # label, timestamp, frame_idx, left_present, left_handedness, left_coords..., right_present, right_handedness, right_coords...
    header = ["label","timestamp","frame_idx"]
    header += ["left_present","left_handedness"]
    header += ["l_" + c for c in for_side_header]
    header += ["right_present","right_handedness"]
    header += ["r_" + c for c in for_side_header]
    return header

def guess_hand_order(multi_handedness):
    # returns list of handedness labels in same order as multi_hand_landmarks
    # multi_handedness is a list of Classification objects (if available)
    if not multi_handedness:
        return []
    res = []
    for hh in multi_handedness:
        try:
            label = hh.classification[0].label  # "Left" or "Right"
        except Exception:
            label = None
        res.append(label)
    return res

def extract_normalized_landmarks(hand_landmarks):
    # returns list of 21 (x,y) normalized floats
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append((float(lm.x), float(lm.y)))
    return pts

def flatten_coords(pts):
    # pts: list of (x,y) pairs length 21
    flat = []
    for x,y in pts:
        flat += [x,y]
    return flat

def pad_hand_data(present, handedness, coords):
    # present: 0/1, handedness: "Left"/"Right"/None, coords: flattened list length 42 or []
    if present:
        return [1, handedness[:1] if handedness else "U"] + coords
    else:
        return [0, "U"] + [0.0]*42

def draw_overlay(img, label, count):
    text = f"Label: {label}  Collected: {count}  (r=record, l=change label, s=save crop, q=quit)"
    cv.putText(img, text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return img

def main(save_crops=False):
    ensure_out_dirs(save_crops)
    header = make_csv_header()
    out_path = Path(OUTPUT_CSV)

    # prepare CSV file (create with header if not exists)
    new_file = not out_path.exists()
    csvfile = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csvfile)
    if new_file:
        writer.writerow(header)
        csvfile.flush()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=MAX_HANDS,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: cannot open webcam")
        return

    label = input("Enter initial label (A, B, 0..9, space for separator): ").strip() or "UNK"
    collected = 0
    frame_idx = 0
    last_key = None

    print("Press 'r' to record a sample, 'l' to change label, 's' to save crop, 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        image = cv.flip(frame, 1)
        h, w = image.shape[:2]
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(rgb)
        debug = image.copy()

        left_present = 0; right_present = 0
        left_handness = "U"; right_handness = "U"
        left_coords = []
        right_coords = []

        if results.multi_hand_landmarks:
            # the repo returns hands in detection order; use handedness to place them as left/right slots
            handedness_list = guess_hand_order(results.multi_handedness)  # same order as landmarks
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(debug, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                pts = extract_normalized_landmarks(hand_landmarks)
                hand_label = handedness_list[idx] if idx < len(handedness_list) else None
                if hand_label and hand_label.lower().startswith("left"):
                    left_present = 1
                    left_handness = "L"
                    left_coords = flatten_coords(pts)
                elif hand_label and hand_label.lower().startswith("right"):
                    right_present = 1
                    right_handness = "R"
                    right_coords = flatten_coords(pts)
                else:
                    # unknown: assign to the first empty slot (prefer left)
                    if left_present == 0:
                        left_present = 1
                        left_handness = hand_label[:1] if hand_label else "U"
                        left_coords = flatten_coords(pts)
                    else:
                        right_present = 1
                        right_handness = hand_label[:1] if hand_label else "U"
                        right_coords = flatten_coords(pts)
        # draw overlay
        debug = draw_overlay(debug, label, collected)
        cv.imshow("Recorder", debug)

        key = cv.waitKey(1) & 0xFF
        if key != 255:
            last_key = key

        if last_key == ord('q') or last_key == 27:
            print("Exiting.")
            break
        elif last_key == ord('l'):
            # change label
            new_label = input("New label: ").strip() or label
            print(f"Label changed: {label} -> {new_label}")
            label = new_label
            last_key = None
        elif last_key == ord('r'):
            # write CSV row
            ts = time.time()
            # ensure coords padded to length 42 each side
            left_part = pad_hand_data(left_present, left_handness, left_coords if left_coords else [])
            right_part = pad_hand_data(right_present, right_handness, right_coords if right_coords else [])
            row = [label, ts, frame_idx] + left_part + right_part
            writer.writerow(row)
            csvfile.flush()
            collected += 1
            print(f"Recorded sample #{collected} label={label} left_present={left_present} right_present={right_present}")
            last_key = None
        elif last_key == ord('s'):
            # save crop image (center crop or bbox around hands)
            # simple center crop for now
            crop = image.copy()
            fname = f"{SAVE_CROPS_DIR}/{label}_{int(time.time())}_{frame_idx}.jpg"
            cv.imwrite(fname, crop)
            print("Saved crop:", fname)
            last_key = None
        # loop continues

    csvfile.close()
    cap.release()
    cv.destroyAllWindows()
    print(f"Done. Total recorded: {collected}. CSV: {OUTPUT_CSV}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_crops", action="store_true", help="Save full-frame crops when 's' pressed")
    args = ap.parse_args()
    main(save_crops=args.save_crops)
