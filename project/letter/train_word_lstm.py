# scripts/train_word_lstm.py
"""
Train an LSTM classifier for word sequences saved in data/word_dataset.

Expected input directory layout (output of prepare_word_dataset.py):
  data/word_dataset/
    X_train.npy  (N_train, seq_len, feat_dim)
    y_train.npy
    X_val.npy
    y_val.npy
    scaler_and_map.joblib  (contains mean,std,labels_map)
    labels_map.json

Usage example:
python .\scripts\train_word_lstm.py --data_dir data/word_dataset --out models/word_model --epochs 50 --batch 64 --lr 1e-3 --hidden 256 128 --dropout 0.4 --device cuda

"""
import argparse
from pathlib import Path
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import time
import os
from tqdm import tqdm
import random

# ---------------- Dataset ----------------
class SequenceDataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        assert len(self.X) == len(self.y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # return (seq_len, feat) float32 tensor and label long tensor
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y

# ---------------- Model ----------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.4, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # first LSTM layer
        h0 = hidden_dims[0]
        self.lstm = nn.LSTM(input_dim, h0, num_layers=1, batch_first=True, bidirectional=bidirectional)
        # optional further FC layers
        fc_layers = []
        prev = h0 * self.num_directions
        for h in hidden_dims[1:]:
            fc_layers.append(nn.Linear(prev, h))
            fc_layers.append(nn.ReLU(inplace=True))
            fc_layers.append(nn.Dropout(dropout))
            prev = h
        fc_layers.append(nn.Linear(prev, num_classes))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)  # out: (B, T, H * num_directions)
        # take last timestep
        last = out[:, -1, :]
        logits = self.fc(last)
        return logits

# ---------------- Utilities ----------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_class_weights(y):
    # returns tensor weights for CrossEntropyLoss
    from collections import Counter
    cnt = Counter(y.tolist())
    classes = sorted(cnt.keys())
    freqs = np.array([cnt[c] for c in classes], dtype=np.float32)
    # weight inversely proportional to frequency
    weights = freqs.sum() / (len(classes) * freqs)
    return torch.from_numpy(weights).float()

def evaluate(model, loader, device, labels_map):
    model.eval()
    ys_true = []
    ys_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            ys_pred.extend(preds.tolist())
            ys_true.extend(yb.numpy().tolist())
    report = classification_report(ys_true, ys_pred, target_names=[k for k,v in sorted(labels_map.items(), key=lambda kv: kv[1])], zero_division=0)
    cm = confusion_matrix(ys_true, ys_pred)
    acc = (np.array(ys_true) == np.array(ys_pred)).mean()
    return acc, report, cm, np.array(ys_true), np.array(ys_pred)

# ---------------- Training Loop ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/word_dataset", help="prepared dataset dir")
    parser.add_argument("--out", default="models/word_model", help="output model dir")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256,128], help="hidden dims for MLP after LSTM")
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--device", choices=["cpu","cuda"], default="cuda")
    parser.add_argument("--patience", type=int, default=6, help="early stopping patience on val acc")
    parser.add_argument("--resume", default="", help="path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="use AMP (automatically enabled when device=cuda)")
    args = parser.parse_args()

    seed_everything(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load dataset files
    X_train_p = Path(args.data_dir) / "X_train.npy"
    y_train_p = Path(args.data_dir) / "y_train.npy"
    X_val_p = Path(args.data_dir) / "X_val.npy"
    y_val_p = Path(args.data_dir) / "y_val.npy"
    scaler_p = Path(args.data_dir) / "scaler_and_map.joblib"
    labels_json = Path(args.data_dir) / "labels_map.json"

    assert X_train_p.exists() and y_train_p.exists(), "Missing train files in " + str(args.data_dir)
    assert X_val_p.exists() and y_val_p.exists(), "Missing val files in " + str(args.data_dir)
    assert scaler_p.exists(), "Missing scaler_and_map.joblib in " + str(args.data_dir)

    labels_map = joblib.load(scaler_p).get("labels_map", None)
    if labels_map is None and labels_json.exists():
        labels_map = json.load(open(labels_json,'r',encoding='utf-8'))
    assert labels_map is not None, "Cannot find labels_map in scaler_or json."

    # dataset dims
    X_train = np.load(X_train_p)
    seq_len, feat_dim = X_train.shape[1], X_train.shape[2]
    num_classes = len(labels_map)
    print("Dataset shapes:", "X_train", X_train.shape, "num_classes", num_classes)

    # Datasets & loaders
    train_ds = SequenceDataset(X_train_p, y_train_p)
    val_ds = SequenceDataset(X_val_p, y_val_p)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("Using device:", device)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=max(0,args.workers//2), pin_memory=(device.type=="cuda"))

    # model
    model = LSTMClassifier(feat_dim, args.hidden, num_classes, dropout=args.dropout, bidirectional=True).to(device)
    print(model)

    # class weights
    y_train = np.load(y_train_p)
    class_weights = compute_class_weights(torch.from_numpy(y_train))
    print("Class weights:", class_weights.numpy())
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler
    use_amp = (device.type=="cuda") and args.amp
    scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_val = 0.0
    patience = args.patience
    patience_cnt = 0

    # resume if checkpoint provided
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck.get("optimizer_state", optimizer.state_dict()))
        start_epoch = ck.get("epoch", 1) + 1
        best_val = ck.get("best_val", 0.0)
        print("Resumed from", args.resume, "start_epoch", start_epoch, "best_val", best_val)

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            pbar.set_postfix(loss=running_loss/total, acc=correct/total)

        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                running_loss += float(loss.item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        val_loss = running_loss / total
        val_acc = correct / total

        elapsed = time.time() - t0
        print(f"Epoch {epoch} - train_loss:{train_loss:.4f} train_acc:{train_acc:.4f} val_loss:{val_loss:.4f} val_acc:{val_acc:.4f} time:{elapsed:.1f}s")

        # checkpoint last
        ck = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "best_val": best_val
        }
        torch.save(ck, out_dir / "last.pt")

        # save best
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            torch.save(ck, out_dir / "best.pt")
            print("Saved best model ->", out_dir / "best.pt")
            patience_cnt = 0
        else:
            patience_cnt += 1
            print("Patience:", patience_cnt, "/", patience)

        # early stopping
        if patience_cnt >= patience:
            print(f"Early stopping at epoch {epoch} (patience {patience}).")
            break

    # load best model and final evaluation
    best_ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ck["model_state"])
    model.to(device).eval()

    val_acc, report, cm, y_true, y_pred = evaluate(model, val_loader, device, labels_map)
    print("Final val acc:", val_acc)
    print("Classification report:\n", report)
    # save final model (package state + meta)
    final_path = out_dir / "final_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "labels_map": labels_map,
        "meta": {"seq_len": seq_len, "feat_dim": feat_dim}
    }, final_path)
    print("Saved final model:", final_path)

    # save confusion matrix csv
    try:
        import pandas as pd
        idx2label = [k for k,v in sorted(labels_map.items(), key=lambda kv: kv[1])]
        cm_df = pd.DataFrame(cm, index=idx2label, columns=idx2label)
        cm_df.to_csv(out_dir / "confusion_matrix.csv")
    except Exception:
        pass

    print("Done training.")

if __name__ == "__main__":
    main()
