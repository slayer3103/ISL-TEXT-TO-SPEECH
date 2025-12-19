#!/usr/bin/env python3
# scripts/get_misclassified.py
"""
List misclassified sequence files and save CSV with true/pred/prob.

Usage:
  python .\word\7_get_misclassified.py --model models/word_lstm/best.pt --data_dir data/word_sequences_ready --out_dir models/word_lstm/eval_out --device cuda
"""
import argparse, numpy as np, torch, csv
from pathlib import Path

def load_seq(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    return d['X'].astype(np.float32), d['mask'].astype(np.uint8), d.get('meta', None)

def find_files(data_dir):
    p = Path(data_dir)
    files = sorted([str(x) for x in p.glob("*/*.npz")])
    labels = sorted(list({Path(f).parent.name for f in files}))
    return files, labels

def build_model(D, nclass, ck, device):
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
    model = BiLSTM(D, hidden=256, nlayers=2, nclass=len(ck["labels"]), dropout=0.3)
    model.load_state_dict(ck["model_state"])
    model = model.to(device)
    model.eval()
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files, labels = find_files(args.data_dir)
    if len(files) == 0:
        print("No files"); return
    label_to_idx = {l:i for i,l in enumerate(labels)}
    idx_to_label = {i:l for l,i in label_to_idx.items()}

    ck = torch.load(args.model, map_location="cpu")
    # get model label order from checkpoint
    ck_labels = list(ck.get("labels", ck.get("labels_map", labels)))
    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")

    # infer input dim
    X0, M0, _ = load_seq(files[0])
    T,D = X0.shape

    model = build_model(D, len(ck_labels), ck, device)

    outp_dir = Path(args.out_dir) if args.out_dir else Path(".")
    outp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outp_dir/"misclassifications.csv"

    rows = []
    with torch.no_grad():
        for p in files:
            X, mask, meta = load_seq(p)
            xt = torch.from_numpy(X).unsqueeze(0).to(device)
            mt = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits = model(xt, mt)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            pred_label = ck_labels[pred_idx] if pred_idx < len(ck_labels) else str(pred_idx)
            # actual label
            true_label = Path(p).parent.name
            prob = float(probs[pred_idx])
            if pred_label != true_label:
                rows.append([p, true_label, pred_label, f"{prob:.4f}"])
    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["npz","true_label","pred_label","pred_prob"])
        for r in rows:
            w.writerow(r)
    print("Wrote", csv_path, "misclassifications:", len(rows))

if __name__ == "__main__":
    main()
