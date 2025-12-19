#!/usr/bin/env python3
"""
Train a BiLSTM word classifier with optional on-the-fly augmentations.

Usage:
  python .\word\5_train_word_lstm_aug.py --data_dir data/word_sequences_ready --out models/word_lstm_aug --epochs 40 --batch 32 --lr 1e-3 --use_augment

Notes:
- Augmentations are only applied to training examples (not val/test).
- Augmentations operate on the per-frame features produced by convert_npz_to_sequences.py.
"""
import argparse, random, numpy as np, torch, os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

# ---------- Augmentation helpers (operate on numpy X arrays shape (T,D)) ----------
def augment_random_crop_scale_translate(X, prob=0.7, scale_range=(0.94,1.06), trans_range=(-0.02,0.02)):
    # Apply small scale and translation to coordinate columns (first 102 dims: left(42),right(42),pose(18))
    if random.random() > prob:
        return X
    X2 = X.copy()
    if X2.shape[1] < 102:
        return X2
    scale = random.uniform(*scale_range)
    tx = random.uniform(*trans_range)
    ty = random.uniform(*trans_range)
    # apply to x (even indices), y (odd indices) across the first 102 dims
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
        X2[i] = X2[max(0, i-1)]  # duplicate previous frame (simple)
    return X2

def default_augment_pipeline(X):
    # Apply a pipeline of augmentations
    X1 = augment_random_crop_scale_translate(X, prob=0.9)
    X1 = augment_jitter_noise(X1, prob=0.9, sigma=0.012)
    X1 = augment_temporal_dropout(X1, prob=0.4)
    return X1

# ---------- Dataset (loads seq .npz). Optionally applies augment_fn in __getitem__ ----------
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, files, label_to_idx=None, augment_fn=None):
        self.files = files
        self.labels = [Path(f).parent.name for f in files]
        if label_to_idx is None:
            self.label_to_idx = {l:i for i,l in enumerate(sorted(set(self.labels)))}
        else:
            self.label_to_idx = label_to_idx
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p, allow_pickle=True)
        X = d['X'].astype(np.float32)    # (T,D)
        mask = d['mask'].astype(np.uint8)
        label = Path(p).parent.name
        y = int(self.label_to_idx[label])
        # apply augmentation (only used for train dataset)
        if self.augment_fn is not None:
            X = self.augment_fn(X)
        return torch.from_numpy(X), torch.from_numpy(mask), y

def collate_fn(batch):
    Xs, masks, ys = zip(*batch)
    X = torch.stack(Xs)      # (B,T,D)
    M = torch.stack(masks)   # (B,T)
    y = torch.tensor(ys, dtype=torch.long)
    return X, M, y

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
        # val split 12% from train
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
    """
    Evaluate model and return (acc, classification_report_str, confusion_matrix).
    Uses explicit `labels=` so sklearn won't error when some classes are missing
    from the current evaluation set.
    """
    model.eval()
    preds = []; trues = []
    with torch.no_grad():
        for X, M, y in loader:
            X = X.to(device); M = M.to(device)
            logits = model(X, M)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist()); trues.extend(y.numpy().tolist())

    # diagnostic: which label indices actually appeared
    unique_labels_in_data = sorted(set(trues) | set(preds))
    if len(unique_labels_in_data) != len(idx_to_label):
        print(f"[evaluate] Warning: only {len(unique_labels_in_data)} unique labels seen in this evaluation "
              f"but idx_to_label contains {len(idx_to_label)} labels. "
              f"Seen indices: {unique_labels_in_data}")

    # force labels ordering to the full range of known classes (0..nclass-1)
    nclass = len(idx_to_label)
    labels = list(range(nclass))
    target_names = [idx_to_label[i] for i in labels]

    # pass labels explicitly so sklearn won't complain if some labels are absent
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(trues, preds, labels=labels, target_names=target_names, zero_division=0)
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

    # datasets
    augment_fn = default_augment_pipeline if args.use_augment else None
    train_ds = SeqDataset(train_files, label_to_idx=label_to_idx, augment_fn=augment_fn)
    val_ds = SeqDataset(val_files, label_to_idx=label_to_idx, augment_fn=None)
    test_ds = SeqDataset(test_files, label_to_idx=label_to_idx, augment_fn=None)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    # peek shape
    sample = np.load(train_files[0], allow_pickle=True)
    T, D = sample['X'].shape
    nclass = len(labels)
    model = BiLSTM(D, hidden=256, nlayers=2, nclass=nclass, dropout=0.3).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn)

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
        train_loss = tot_loss / tot
        train_acc = correct / tot
        val_acc, val_rep, _ = evaluate(model, val_loader, dev, idx_to_label)
        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        # save best
        if val_acc > best_val + 1e-6:
            best_val = val_acc
            ck = {"model_state": model.state_dict(), "labels": labels}
            torch.save(ck, outp/"best.pt")
            print("Saved best model.")
    print("Training done. Best val acc:", best_val)

    # final test evaluation if test files exist
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
