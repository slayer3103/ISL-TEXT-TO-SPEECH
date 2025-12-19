#!/usr/bin/env python3
# scripts/eval_sequences.py
"""
Evaluate a saved BiLSTM model on sequence .npz files.

Usage:
  python .\word\6_eval_sequences.py --model models/word_lstm_aug/best.pt --data_dir data/word_sequences_ready --out_dir models/word_lstm_aug/eval_out --device cuda
"""
import argparse, numpy as np, torch, os, csv, math
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

def find_files(data_dir):
    p = Path(data_dir)
    files = sorted([str(x) for x in p.glob("*/*.npz")])
    labels = sorted(list({Path(f).parent.name for f in files}))
    return files, labels

def load_model(model_path, device):
    ck = torch.load(model_path, map_location="cpu")
    labels = ck.get("labels", None) or ck.get("labels_map", None)
    if labels is None:
        raise RuntimeError("Model checkpoint has no 'labels' or 'labels_map' entry.")
    model_state = ck["model_state"]
    # we need model architecture; reconstruct same BiLSTM as training
    import torch.nn as nn
    class BiLSTM(nn.Module):
        def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, nclass))
        def forward(self,x,mask):
            out, _ = self.lstm(x)
            mask = mask.unsqueeze(-1).float()
            summed = (out * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-3)
            pooled = summed / denom
            logits = self.head(pooled)
            return logits
    # find input dim from example file later; temporarily return state and labels
    return model_state, labels

def load_seq(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    X = d['X'].astype(np.float32)
    mask = d['mask'].astype(np.uint8)
    meta = d['meta'].item() if 'meta' in d.files else {}
    return X, mask, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)  # will search subfolders
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files, labels = find_files(args.data_dir)
    if len(files) == 0:
        print("No sequence files found under", args.data_dir); return
    label_to_idx = {l:i for i,l in enumerate(labels)}
    idx_to_label = {i:l for l,i in label_to_idx.items()}

    model_state, labels_in_ck = load_model(args.model, args.device)

    # infer input dim from first file
    X0, M0, meta0 = load_seq(files[0])
    T, D = X0.shape
    nclass = len(labels)
    # rebuild model
    import torch.nn as nn
    class BiLSTM(nn.Module):
        def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
            self.head = nn.Sequential(nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, nclass))
        def forward(self,x,mask):
            out, _ = self.lstm(x)
            mask = mask.unsqueeze(-1).float()
            summed = (out * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-3)
            pooled = summed / denom
            logits = self.head(pooled)
            return logits

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    model = BiLSTM(D, hidden=256, nlayers=2, nclass=nclass, dropout=0.3).to(device)
    # Map labels if checkpoint has labels mapping (best.pt labels list order)
    # We'll assume ck["labels"] is a list in index order
    ck = torch.load(args.model, map_location="cpu")
    ck_labels = ck.get("labels")
    if ck_labels:
        # ck_labels is list of label names in model order -> build mapping from model idx -> global idx
        model_label_order = list(ck_labels)
        # build model->global mapping
        model_to_global = [label_to_idx[l] if l in label_to_idx else None for l in model_label_order]
    else:
        model_to_global = None

    model.load_state_dict(ck["model_state"])
    model.eval()

    y_true=[]; y_pred=[]; paths=[]
    with torch.no_grad():
        for p in files:
            X, mask, meta = load_seq(p)
            label = Path(p).parent.name
            y_true.append(label_to_idx[label])
            xt = torch.from_numpy(X).unsqueeze(0).to(device)  # (1,T,D)
            mt = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits = model(xt, mt)
            pred = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
            # if model had ck_labels mapping, map pred to global idx
            if model_to_global:
                mapped = model_to_global[pred]
                if mapped is None:
                    # unknown label mapping
                    pred_global = pred
                else:
                    pred_global = mapped
            else:
                pred_global = pred
            y_pred.append(pred_global)
            paths.append(p)

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    overall_acc = (y_true==y_pred).mean()
    print(f"Overall accuracy: {overall_acc:.6f}\n")

    # per-class metrics (sklearn)
    labels_sorted = [idx_to_label[i] for i in range(len(idx_to_label))]
    rep = classification_report(y_true, y_pred, labels=list(range(len(idx_to_label))), target_names=labels_sorted, zero_division=0)
    print("Classification report (sklearn):\n")
    print(rep)

    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(idx_to_label))), zero_division=0)
    # save per-class csv
    if args.out_dir:
        outp = Path(args.out_dir); outp.mkdir(parents=True, exist_ok=True)
        csvp = outp / "per_class_metrics.csv"
        with open(csvp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label","precision","recall","f1","support"])
            for i,lab in enumerate(labels_sorted):
                w.writerow([lab, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(support[i])])
        print("Per-class metrics saved to:", csvp)

        # confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(idx_to_label))))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_xticks(np.arange(len(labels_sorted))); ax.set_yticks(np.arange(len(labels_sorted)))
        ax.set_xticklabels(labels_sorted, rotation=90, fontsize=8)
        ax.set_yticklabels(labels_sorted, fontsize=8)
        ax.set_ylabel("True"); ax.set_xlabel("Pred")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = outp / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        print("Confusion matrix saved to:", cm_path)

    # print worst-performing classes by f1
    worst_idx = np.argsort(f1)[:10]
    print("\nWorst-performing classes (by F1):")
    print("rank label f1 support")
    for r,i in enumerate(worst_idx, start=1):
        print(f"{r:2d} {labels_sorted[i]:20s} {f1[i]:.4f} {int(support[i])}")

if __name__ == "__main__":
    main()
