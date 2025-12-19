#!/usr/bin/env python3
"""
Train a BiLSTM word classifier with robust .npz loading and consistent feature dims.

Usage:
  python .\word\5_train_word_lstm_aug.py --data_dir data/word_sequences_ready --out models/word_lstm_aug --epochs 40 --batch 32 --lr 1e-3 --use_augment

Changes vs your previous file:
 - robust .npz loader (tries many key names, falls back to first array)
 - enforces consistent per-frame feature dimension (EXPECTED_D, default 109) by zero-padding/truncation
 - collate function pads time dimension (T) to max in batch
 - no extra debug/log file dumps
"""
import argparse
import random
import numpy as np
import torch
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# Expected per-frame feature dimension used by model/inference (left42 + right42 + pose18 + 2vel + 1dist + present2 = 109)
EXPECTED_D = 109

# ---------- Augmentation helpers (operate on numpy X arrays shape (T,D)) ----------
def augment_random_crop_scale_translate(X, prob=0.7, scale_range=(0.94,1.06), trans_range=(-0.02,0.02)):
    if random.random() > prob:
        return X
    X2 = X.copy()
    if X2.shape[1] < 102:
        return X2
    scale = random.uniform(*scale_range)
    tx = random.uniform(*trans_range)
    ty = random.uniform(*trans_range)
    for offset in range(0, 102, 2):
        X2[:, offset] = X2[:, offset] * scale + tx
        X2[:, offset+1] = X2[:, offset+1] * scale + ty
    return X2

def augment_jitter_noise(X, prob=0.6, sigma=0.01):
    if random.random() > prob:
        return X
    X2 = X.copy()
    if X2.shape[1] < 102:
        return X2
    noise = np.random.normal(0, sigma, size=X2[:, :102].shape).astype(np.float32)
    X2[:, :102] += noise
    return X2

def augment_temporal_dropout(X, prob=0.5, max_drop=3):
    if random.random() > prob:
        return X
    X2 = X.copy()
    T = X2.shape[0]
    drops = random.randint(1, max_drop)
    for _ in range(drops):
        i = random.randint(0, T-1)
        X2[i] = X2[max(0, i-1)]
    return X2

def default_augment_pipeline(X):
    X1 = augment_random_crop_scale_translate(X, prob=0.9)
    X1 = augment_jitter_noise(X1, prob=0.9, sigma=0.012)
    X1 = augment_temporal_dropout(X1, prob=0.4)
    return X1

# ---------- Robust npz loader ----------
def load_npz_flexible(path, expected_D=EXPECTED_D):
    """
    Load an npz file robustly:
     - Try common key names for X, mask, meta/y
     - Fall back to first array if X not found
     - Ensure X has shape (T, expected_D) by padding or truncating columns
     - Ensure mask is length T (create ones if missing)
     - Return (X (np.float32), mask (np.uint8), label_meta dict)
    """
    d = np.load(path, allow_pickle=True)
    files = list(d.files)

    def pick(cands):
        for k in cands:
            if k in d:
                return d[k]
        return None

    X = pick(['X', 'feats_arr', 'feats', 'features', 'arr_0', '0'])
    mask = pick(['M', 'mask', 'm', 'arr_1', '1'])
    meta = pick(['meta', 'info', 'meta_dict', 'dict', 'arr_2', '2'])
    label = None

    if X is None:
        if len(files) == 0:
            raise RuntimeError(f"Empty npz file: {path}")
        # fallback to first array
        X = d[files[0]]

    X = np.array(X, dtype=np.float32)
    T = X.shape[0]

    # ensure mask
    if mask is None:
        mask = np.ones((T,), dtype=np.uint8)
    else:
        mask = np.array(mask, dtype=np.uint8)
        # if mask length mismatches, try to adapt
        if mask.shape[0] != T:
            if mask.shape[0] > T:
                mask = mask[:T]
            else:
                # pad with ones
                pad_len = T - mask.shape[0]
                mask = np.concatenate([mask, np.ones(pad_len, dtype=np.uint8)], axis=0)

    # normalize feature width to expected_D
    D = X.shape[1] if X.ndim > 1 else 1
    if D < expected_D:
        pad_width = expected_D - D
        pad = np.zeros((T, pad_width), dtype=np.float32)
        X = np.concatenate([X, pad], axis=1)
    elif D > expected_D:
        X = X[:, :expected_D]

    # ensure 2D
    if X.ndim == 1:
        X = X[:, None]

    # return X (T,expected_D), mask (T,), meta dict if available
    meta_dict = {}
    if meta is not None:
        try:
            # if meta is saved as array-object containing dict
            if isinstance(meta, np.ndarray) and meta.dtype == object and meta.size == 1:
                meta_dict = dict(meta.item()) if isinstance(meta.item(), dict) else {}
            elif isinstance(meta, dict):
                meta_dict = meta
        except Exception:
            meta_dict = {}
    return X.astype(np.float32), mask.astype(np.uint8), meta_dict

# ---------- Dataset ----------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, files, label_to_idx=None, augment_fn=None, expected_D=EXPECTED_D):
        self.files = files
        # derive labels from parent folder names
        self.labels = [Path(f).parent.name for f in files]
        if label_to_idx is None:
            self.label_to_idx = {l:i for i,l in enumerate(sorted(set(self.labels)))}
        else:
            self.label_to_idx = label_to_idx
        self.augment_fn = augment_fn
        self.expected_D = expected_D

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        X, mask, meta = load_npz_flexible(p, expected_D=self.expected_D)
        label_name = Path(p).parent.name
        y = int(self.label_to_idx[label_name])
        if self.augment_fn is not None:
            X = self.augment_fn(X)
            # ensure augment didn't change width
            if X.shape[1] != self.expected_D:
                # pad/truncate as in loader
                D = X.shape[1]
                T = X.shape[0]
                if D < self.expected_D:
                    pad = np.zeros((T, self.expected_D - D), dtype=np.float32)
                    X = np.concatenate([X, pad], axis=1)
                else:
                    X = X[:, :self.expected_D]
        # ensure types
        return torch.from_numpy(X.astype(np.float32)), torch.from_numpy(mask.astype(np.uint8)), y

# ---------- Collate (pads time dimension T to max in batch) ----------
def collate_fn_padded(batch):
    """
    Batch is list of (X (T,D) torch), mask (T,) torch, y int)
    Pads sequences in time to the max T in batch; D must be consistent across batch.
    Returns X (B, T_max, D), M (B, T_max), y (B,)
    """
    Xs, Ms, ys = zip(*batch)
    Ts = [x.shape[0] for x in Xs]
    maxT = max(Ts)
    B = len(Xs)
    D = Xs[0].shape[1]
    # verify D consistent
    for x in Xs:
        if x.shape[1] != D:
            raise ValueError(f"Inconsistent feature dimension in batch: expected D={D}, got {x.shape[1]}")
    X_buf = torch.zeros((B, maxT, D), dtype=Xs[0].dtype)
    M_buf = torch.zeros((B, maxT), dtype=Ms[0].dtype)
    y_buf = torch.tensor(ys, dtype=torch.long)
    for i, (x, m, _) in enumerate(batch):
        t = x.shape[0]
        X_buf[i, :t, :] = x
        M_buf[i, :t] = m
    return X_buf, M_buf, y_buf

# ---------- Model ----------
class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden*2),
                                        torch.nn.Linear(hidden*2, 128),
                                        torch.nn.ReLU(),
                                        torch.nn.Dropout(dropout),
                                        torch.nn.Linear(128, nclass))
    def forward(self,x,mask):
        out, _ = self.lstm(x)
        maskf = mask.unsqueeze(-1).float()  # (B,T,1)
        summed = (out * maskf).sum(dim=1)
        denom = maskf.sum(dim=1).clamp(min=1e-3)
        pooled = summed / denom
        logits = self.head(pooled)
        return logits

# ---------- Utilities ----------
def find_seq_files(data_dir):
    p = Path(data_dir)
    files = sorted([str(x) for x in p.glob("*/*.npz")])
    return files

def split_files(files, holdout_signers):
    if holdout_signers:
        train_files=[]; val_files=[]; test_files=[]
        for f in files:
            d = np.load(f, allow_pickle=True)
            meta = d['meta'].item() if 'meta' in d.files else {}
            orig = meta.get('orig_npz', '') or ''
            if any(h in orig for h in holdout_signers) or any(h in f for h in holdout_signers):
                test_files.append(f)
            else:
                train_files.append(f)
        random.shuffle(train_files)
        n_val = max(1, int(0.12 * len(train_files)))
        val_files = train_files[:n_val]
        train_files = train_files[n_val:]
        return train_files, val_files, test_files
    else:
        random.shuffle(files)
        ntest = max(1, int(0.12 * len(files)))
        nval = max(1, int(0.12 * (len(files)-ntest)))
        test = files[:ntest]
        val = files[ntest:ntest+nval]
        train = files[ntest+nval:]
        return train, val, test

def evaluate(model, loader, device, idx_to_label):
    model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for X, M, y in loader:
            X = X.to(device); M = M.to(device)
            logits = model(X, M)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist()); trues.extend(y.numpy().tolist())
    labels = list(range(len(idx_to_label)))
    report = classification_report(trues, preds, labels=labels, target_names=[idx_to_label[i] for i in labels], zero_division=0)
    cm = confusion_matrix(trues, preds, labels=labels)
    acc = (np.array(trues) == np.array(preds)).mean() if len(trues) > 0 else 0.0
    return acc, report, cm

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout_signers", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_augment", action="store_true", help="enable augmentations during training")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    files = find_seq_files(args.data_dir)
    print("Found sequence files:", len(files))
    train_files, val_files, test_files = split_files(files, args.holdout_signers)
    print("Train/Val/Test counts:", len(train_files), len(val_files), len(test_files))

    labels = sorted(list({Path(f).parent.name for f in files}))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    idx_to_label = {i:l for l,i in label_to_idx.items()}

    augment_fn = default_augment_pipeline if args.use_augment else None
    train_ds = SeqDataset(train_files, label_to_idx=label_to_idx, augment_fn=augment_fn, expected_D=EXPECTED_D)
    val_ds = SeqDataset(val_files, label_to_idx=label_to_idx, augment_fn=None, expected_D=EXPECTED_D)
    test_ds = SeqDataset(test_files, label_to_idx=label_to_idx, augment_fn=None, expected_D=EXPECTED_D)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    # peek shape using robust loader (not assuming exact key names)
    sample_X, _, _ = load_npz_flexible(train_files[0], expected_D=EXPECTED_D)
    T, D = sample_X.shape
    nclass = len(labels)
    model = BiLSTM(D, hidden=256, nlayers=2, nclass=nclass, dropout=0.3).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn_padded, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn_padded, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn_padded, num_workers=2)

    best_val = 0.0
    outp = Path(args.out); outp.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss=0; tot=0; correct=0
        for X, M, y in train_loader:
            X = X.to(dev); M = M.to(dev); y = y.to(dev)
            opt.zero_grad()
            logits = model(X, M)
            loss = criterion(logits, y)
            loss.backward(); opt.step()
            tot_loss += float(loss.item()) * X.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred==y).sum().item()
            tot += X.size(0)
        train_loss = tot_loss / tot if tot>0 else 0.0
        train_acc = correct / tot if tot>0 else 0.0
        val_acc, val_rep, _ = evaluate(model, val_loader, dev, idx_to_label)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            ck = {"model_state": model.state_dict(), "labels": labels}
            torch.save(ck, outp/"best.pt")
            print("Saved best model.")
    print("Training done. Best val acc:", best_val)

    if len(test_files) > 0:
        ck = torch.load(outp/"best.pt", map_location=dev)
        model.load_state_dict(ck["model_state"])
        test_acc, test_rep, test_cm = evaluate(model, test_loader, dev, idx_to_label)
        print("Test acc:", test_acc)
        print("Test classification report:\n", test_rep)
        torch.save({"model_state": model.state_dict(), "labels": labels}, outp/"final_model.pt")
        print("Saved final model.")

if __name__ == "__main__":
    main()
