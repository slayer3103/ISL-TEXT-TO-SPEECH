# utils/landmark_dataset.py
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

def read_csv_rows(csv_path: str) -> List[List[str]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # assume header exists
        for r in reader:
            if not r:
                continue
            rows.append(r)
    return header, rows

def parse_row_to_feature(row: List[str], header: List[str]) -> Tuple[str, np.ndarray]:
    
    label = row[0]
    # left_present at index 3 (0-based header positions: 0 label,1 ts,2 frame_idx,3 left_present)
    # but safer to find by names
    h = [c.lower() for c in header]
    # find start indices
    try:
        li = h.index("left_present")
        ri = h.index("right_present")
    except ValueError:
        # fallback: assume fixed positions (header created by recorder script)
        li = 3
        ri = 3 + 2 + 42  # 3(label,ts,frame), left_present,left_handness + 42 coords
    # left present flag
    left_present = int(row[li])
    # left coords start at li+2
    left_coords = []
    for i in range(li+2, li+2+42):
        try:
            left_coords.append(float(row[i]))
        except Exception:
            left_coords.append(0.0)
    right_present = int(row[ri])
    right_coords = []
    for i in range(ri+2, ri+2+42):
        try:
            right_coords.append(float(row[i]))
        except Exception:
            right_coords.append(0.0)
    # Feature vector: concatenate left_coords then right_coords (length 84)
    feat = np.array(left_coords + right_coords, dtype=np.float32)
    return label, feat

class LandmarkCSV(Dataset):
    def __init__(self, csv_path: str, labels_map: Optional[dict]=None, transform=None):
        self.csv_path = Path(csv_path)
        self.header, self.rows = read_csv_rows(str(self.csv_path))
        self.samples = []
        for r in self.rows:
            lbl, feat = parse_row_to_feature(r, self.header)
            self.samples.append((lbl, feat))
        # build label mapping if not provided
        if labels_map is None:
            unique = sorted({s[0] for s in self.samples})
            self.labels_map = {lab:i for i,lab in enumerate(unique)}
        else:
            self.labels_map = labels_map
        # convert to final arrays
        self.X = []
        self.y = []
        for lbl, feat in self.samples:
            if lbl not in self.labels_map:
                # skip unknown label
                continue
            self.X.append(feat)
            self.y.append(self.labels_map[lbl])
        self.X = np.stack(self.X).astype(np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        self.transform = transform

    def classes(self):
        inv = {v:k for k,v in self.labels_map.items()}
        return [inv[i] for i in range(len(inv))]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        # return tensor
        return torch.from_numpy(x), int(self.y[idx])
