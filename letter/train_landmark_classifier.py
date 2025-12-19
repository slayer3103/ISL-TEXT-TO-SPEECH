#!/usr/bin/env python
"""
Train a landmark-vector MLP classifier.

Usage example:
python .\scripts\train_landmark_classifier.py --csv data/recorded_keypoints.csv --out models/landmark_model --epochs 30 --batch 64 --workers 2 --use_augment
"""

import sys, pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from utils.landmark_dataset import LandmarkCSV

# -------------- Top-level Dataset wrapper (picklable on Windows) --------------
class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, X, y, mean, std, augment_fn=None):
        self.X = X.astype(np.float32)
        self.y = y
        self.mean = mean
        self.std = std
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment_fn is not None:
            x = self.augment_fn(x)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[idx])


# ---------------- Simple MLP model ----------------
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.4):
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

# ---------------- utility functions ----------------
def compute_scaler(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

# ---------------- main training ----------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print("Device:", device)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # load dataset
    dataset = LandmarkCSV(args.csv)
    num_classes = len(dataset.classes())
    print("Found classes:", dataset.labels_map)
    X = dataset.X  # numpy (N, D)
    y = dataset.y

    # train/val/test split (stratified)
    if len(y) < 10:
        raise RuntimeError("Not enough samples in CSV to perform splits. Collect more data.")
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=args.test_frac, random_state=42, stratify=y)
    idx_train, idx_val = train_test_split(idx_train, test_size=args.val_frac/(1-args.test_frac), random_state=42, stratify=y[idx_train])

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # compute scaler on train and save
    mean, std = compute_scaler(X_train)
    joblib.dump({"mean": mean, "std": std, "labels_map": dataset.labels_map}, Path(args.out)/"scaler_and_map.joblib")
    print("Saved scaler and label map to", Path(args.out)/"scaler_and_map.joblib")

    # optionally import augment pipeline
    augment_fn = None
    if args.use_augment:
        try:
            from utils.landmark_augment import default_augment
            augment_fn = default_augment()
            print("Using augmentations during training.")
        except Exception as e:
            print("Could not import augment module (utils.landmark_augment). Proceeding without augmentation. Error:", e)
            augment_fn = None

    # build datasets
    train_ds = TransformWrapper(X_train, y_train, mean, std, augment_fn=augment_fn)
    val_ds = TransformWrapper(X_val, y_val, mean, std, augment_fn=None)
    test_ds = TransformWrapper(X_test, y_test, mean, std, augment_fn=None)

    # Ensure workers is non-negative
    workers = max(0, int(args.workers))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=(device.type=="cuda") and workers>0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=max(0, workers//2), pin_memory=(device.type=="cuda") and workers>0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=max(0, workers//2), pin_memory=(device.type=="cuda") and workers>0)

    input_dim = X.shape[1]
    model = MLP(input_dim, hidden_dims=[256,128], num_classes=num_classes, dropout=args.dropout).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = 0.0
    # modern GradScaler
    scaler = torch.amp.GradScaler(enabled=(device.type=="cuda"))

    patience_counter = 0
    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        tloss = 0.0; tacc=0; tcount=0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            # autocast with explicit device_type for compatibility
            with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=="cuda")):
                out = model(xb)
                loss = criterion(out, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tloss += float(loss.item()) * xb.size(0)
            preds = out.argmax(dim=1)
            tacc += (preds==yb).sum().item()
            tcount += xb.size(0)
        train_loss = tloss / tcount
        train_acc = tacc / tcount

        # val
        model.eval()
        vloss=0.0; vacc=0; vcount=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda' if device.type=='cuda' else 'cpu', enabled=(device.type=="cuda")):
                    out = model(xb)
                    loss = criterion(out, yb)
                vloss += float(loss.item()) * xb.size(0)
                preds = out.argmax(dim=1)
                vacc += (preds==yb).sum().item()
                vcount += xb.size(0)
        val_loss = vloss / vcount
        val_acc = vacc / vcount

        print(f"Epoch {epoch}/{args.epochs} - train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f}")

        # checkpoint
        ck = {"epoch":epoch, "model_state": model.state_dict(), "mean":mean, "std":std, "labels_map": dataset.labels_map, "val_acc": val_acc}
        torch.save(ck, Path(args.out)/"last.pt")
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            torch.save(ck, Path(args.out)/"best.pt")
            print("Saved best:", Path(args.out)/"best.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping after {epoch} epochs (patience {args.patience}).")
            break

    # final eval on test
    print("Evaluating on test set...")
    ck = torch.load(Path(args.out)/"best.pt", map_location=device)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    y_true=[]; y_pred=[]
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            out = model(xb)
            preds = out.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(yb.numpy().tolist())
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true==y_pred).mean()
    print("Test accuracy:", acc)

    # save final model
    torch.save({"model_state":model.state_dict(), "labels_map": dataset.labels_map, "mean":mean, "std":std}, Path(args.out)/"final_model.pt")
    print("Saved final model to", Path(args.out)/"final_model.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="recorded CSV produced by recorder")
    parser.add_argument("--out", required=True, help="output dir to save checkpoints and scaler")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--val_frac", type=float, default=0.12)
    parser.add_argument("--test_frac", type=float, default=0.12)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--workers", type=int, default=2, help="num data loader workers (set 0 if problems on Windows)")
    parser.add_argument("--use_augment", action="store_true", help="enable augmentations from utils.landmark_augment")
    args = parser.parse_args()
    main(args)
