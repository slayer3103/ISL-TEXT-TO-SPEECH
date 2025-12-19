#!/usr/bin/env python
"""
Evaluate trained landmark MLP on a CSV of recorded landmarks.

Usage:
python .\scripts\eval_landmark_classifier.py --csv data/recorded_keypoints.csv --model models/landmark_model/final_model.pt --out models/landmark_model/eval_out
"""

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import torch
from utils.landmark_dataset import LandmarkCSV
from sklearn.metrics import classification_report, confusion_matrix
import csv

def load_ck(path):
    ck = torch.load(path, map_location="cpu")
    model_state = ck.get("model_state", ck)
    mean = ck.get("mean", None)
    std = ck.get("std", None)
    labels_map = ck.get("labels_map", None)
    return model_state, mean, std, labels_map

def build_model(input_dim, hidden_dims, n_classes):
    import torch.nn as nn
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            prev = input_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev,h)); layers.append(nn.ReLU()); layers.append(nn.Dropout(0.4))
                prev = h
            layers.append(nn.Linear(prev, n_classes))
            self.net = nn.Sequential(*layers)
        def forward(self,x): return self.net(x)
    return MLP()

def save_confusion_csv(cm, labels, outpath):
    outp = Path(outpath)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [""] + labels
        writer.writerow(header)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV of recorded landmarks (same format as recorder)")
    p.add_argument("--model", required=True, help="trained checkpoint (final_model.pt or best.pt)")
    p.add_argument("--out", default=None, help="output folder to save confusion matrix CSV")
    args = p.parse_args()

    # load dataset
    dataset = LandmarkCSV(args.csv)
    X = dataset.X  # numpy (N, D)
    y = dataset.y

    # load checkpoint
    model_state, mean, std, labels_map = load_ck(args.model)

    # if labels_map missing in ck, try scaler file next to model
    if labels_map is None:
        # try joblib next to model
        try:
            import joblib
            job = Path(args.model).parent / "scaler_and_map.joblib"
            if job.exists():
                info = joblib.load(job)
                mean = mean if mean is not None else info.get("mean", None)
                std = std if std is not None else info.get("std", None)
                labels_map = info.get("labels_map", labels_map)
        except Exception:
            pass

    if labels_map is None:
        # fallback to dataset's labels_map built at load time (not ideal)
        labels_map = dataset.labels_map
        print("Warning: labels_map not found in checkpoint; using dataset.labels_map.")

    # ensure labels list ordered by index
    labels = [None] * len(labels_map)
    for k,v in labels_map.items():
        labels[v] = k

    # build model and load state
    input_dim = X.shape[1]
    model = build_model(input_dim, [256,128], len(labels))
    model.load_state_dict(model_state)
    model.eval()

    # apply scaler if present
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        std[std == 0] = 1.0
        Xs = (X - mean) / std
    else:
        print("Warning: mean/std not found in checkpoint; evaluating without scaling.")
        Xs = X

    # run inference (CPU is fine for eval)
    with torch.no_grad():
        logits = model(torch.from_numpy(Xs.astype(np.float32)))
        preds = logits.argmax(dim=1).numpy()

    acc = (preds == y).mean()
    print(f"Accuracy: {acc:.6f}\n")
    print("Classification report:")
    print(classification_report(y, preds, target_names=labels, digits=4))

    cm = confusion_matrix(y, preds)
    print("Confusion matrix shape:", cm.shape)

    if args.out:
        outdir = Path(args.out)
        outdir.mkdir(parents=True, exist_ok=True)
        # save confusion matrix CSV
        save_confusion_csv(cm, labels, outdir / "confusion_matrix.csv")
        # also save per-class counts CSV
        with open(outdir / "per_class_counts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label","support","pred_count"])
            for i, lab in enumerate(labels):
                support = int((y == i).sum())
                pred_count = int((preds == i).sum())
                writer.writerow([lab, support, pred_count])
        print("Saved evaluation outputs to:", outdir)