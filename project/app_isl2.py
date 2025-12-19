
"""

run unsing
  python scripts/app_isl.py --mode infer --model_path models/landmark_model/final_model.pt --scaler_path models/landmark_model/scaler_and_map.joblib

"""

import sys
import os
import argparse
import pathlib
import time
import cv2
import torch
import numpy as np
import mediapipe as mp
import joblib
import csv

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


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


def build_model(input_dim, hidden_dims, num_classes, device, dropout=0.4):
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


def extract_lr_landmarks(results):
    left = [0.0] * 42
    right = [0.0] * 42
    left_p = 0
    right_p = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'Left' or 'Right'
            pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            flat = [v for pair in pts for v in pair]
            if len(flat) == 42:
                if label == "left":
                    left = flat
                    left_p = 1
                elif label == "right":
                    right = flat
                    right_p = 1

    feat = np.array(left + right, dtype=np.float32)
    return feat, left_p, right_p



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and (args.device == "cuda") else "cpu")
    print("Using device:", device)

    # Setup MediaPipe
    mp_h = mp.solutions.hands
    hands = mp_h.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_det,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print("Cannot open camera", args.camera_idx)
        return

    # Logging / collection mode
    csv_file = None
    csv_writer = None
    if args.mode == "collect":
        # open CSV for appending
        csv_file = open(args.out_csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        # write header if empty
        if os.stat(args.out_csv).st_size == 0:
            header = ["label"] + [f"x{i}" for i in range(84)]
            csv_writer.writerow(header)
        print("Collection mode: label =", args.collect_label)

    # Inference mode: load model & scaler
    model = None
    mean = std = None
    labels_map = None
    inv_labels = None
    if args.mode == "infer":
        model_state, mean, std, labels_map = load_model_and_scaler(args.model_path, args.scaler_path, device)
        inv_labels = {v: k for k, v in labels_map.items()}
        model = build_model(84, args.hidden_dims or [256,128], len(labels_map), device, dropout=args.dropout)
        model.load_state_dict(model_state)
        model.eval()
        print("Loaded model, classes:", labels_map)

    print("Press 'q' to quit. In collect mode, press 'c' to capture current frame as sample.")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        feat, lp, rp = extract_lr_landmarks(results)

        # draw landmarks
        if results.multi_hand_landmarks:
            for h in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(img, h, mp_h.HAND_CONNECTIONS)

        if args.mode == "infer":
            # inference
            x = (feat - mean) / std
            x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x_t)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            best = int(np.argmax(probs))
            prob = float(probs[best])
            label = inv_labels.get(best, str(best))
            display = f"{label} ({prob:.2f})"
            cv2.putText(img, display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        elif args.mode == "collect":
            # show label to collect
            cv2.putText(img, f"Collecting: {args.collect_label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

        cv2.imshow("ISL App", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if args.mode == "collect" and key == ord("c"):
            # write feature with label
            if csv_writer:
                csv_writer.writerow([args.collect_label] + feat.tolist())
                print("Saved sample for", args.collect_label)
        # you can add further key controls if needed (pause, next label etc.)

    if csv_file:
        csv_file.close()
    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["infer", "collect"], required=True)
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--min_det", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to run inference on")
    # For inference mode
    parser.add_argument("--model_path", help="path to PyTorch model .pt")
    parser.add_argument("--scaler_path", help="joblib scaler_and_map")
    parser.add_argument("--hidden_dims", nargs="+", type=int, help="hidden dims, e.g. 256 128")
    parser.add_argument("--dropout", type=float, default=0.4)
    # For collection mode
    parser.add_argument("--out_csv", help="CSV file to append collected keypoints")
    parser.add_argument("--collect_label", help="Label to assign samples in collect mode")

    args = parser.parse_args()
    main(args)
