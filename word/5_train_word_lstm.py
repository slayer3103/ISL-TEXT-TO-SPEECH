#!/usr/bin/env python3
# scripts/train_word_lstm.py
"""
Train a BiLSTM word classifier on sequences produced by convert_npz_to_sequences.py.

Usage:
  python .\word\5_train_word_lstm.py --data_dir data/word_sequences_ready --out models/word_lstm --epochs 40 --batch 32 --lr 1e-3 --holdout_signers p05 p06
"""
import argparse, os, numpy as np, torch, random
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.labels = [str(Path(f).parent.name) for f in files]
        self.label_to_idx = {l:i for i,l in enumerate(sorted(set(self.labels)))}
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        d = np.load(p, allow_pickle=True)
        X = d['X'].astype(np.float32)       # (T,D)
        mask = d['mask'].astype(np.uint8)   # (T,)
        meta = d['meta'].item() if 'meta' in d.files else {}
        label = Path(p).parent.name
        y = self.label_to_idx[label]
        return torch.from_numpy(X), torch.from_numpy(mask), y, str(p), meta

def collate_fn(batch):
    Xs, masks, ys, paths, metas = zip(*batch)
    X = torch.stack(Xs)  # (B,T,D)
    M = torch.stack(masks) # (B,T)
    y = torch.tensor(ys, dtype=torch.long)
    return X, M, y, paths, metas

class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden=256, nlayers=2, nclass=10, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden, nlayers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden*2), torch.nn.Linear(hidden*2, 128), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(128, nclass))
    def forward(self,x,mask):
        # x: (B,T,D)
        out, _ = self.lstm(x)
        # pool masked frames -- mean over valid frames
        mask = mask.unsqueeze(-1).float()  # (B,T,1)
        summed = (out * mask).sum(dim=1)   # (B,hidden*2)
        denom = mask.sum(dim=1).clamp(min=1e-3)
        pooled = summed / denom
        logits = self.head(pooled)
        return logits

def find_seq_files(data_dir):
    p = Path(data_dir)
    files = sorted([str(x) for x in p.glob("*/*.npz")])
    return files

def split_files(files, holdout_signers):
    # if holdout_signers provided, use files whose meta.orig_npz path contains signer id into test set.
    if holdout_signers:
        train_files=[]; test_files=[]
        for f in files:
            try:
                d = np.load(f, allow_pickle=True)
                meta = d['meta'].item() if 'meta' in d.files else {}
                orig = meta.get('orig_npz', '')
            except Exception:
                orig = ''
            if any(h in orig for h in holdout_signers):
                test_files.append(f)
            else:
                train_files.append(f)
        # from train_files create val split 12% random
        random.shuffle(train_files)
        nval = max(1, int(0.12 * len(train_files)))
        val_files = train_files[:nval]
        train_files2 = train_files[nval:]
        return train_files2, val_files, test_files
    else:
        # stratified-ish by label: simple shuffle & split
        random.shuffle(files)
        ntest = max(1, int(0.12 * len(files)))
        nval = max(1, int(0.12 * (len(files)-ntest)))
        test_files = files[:ntest]
        val_files = files[ntest:ntest+nval]
        train_files = files[ntest+nval:]
        return train_files, val_files, test_files

def evaluate(model, loader, device, idx_to_label):
    model.eval()
    preds=[]; trues=[]; paths=[]
    with torch.no_grad():
        for X, M, y, pths, metas in loader:
            X=X.to(device); M=M.to(device)
            logits = model(X, M)
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist()); trues.extend(y.tolist()); paths.extend(pths)
    report = classification_report(trues, preds, target_names=[idx_to_label[i] for i in range(len(idx_to_label))], zero_division=0)
    cm = confusion_matrix(trues, preds)
    acc = (np.array(trues)==np.array(preds)).mean()
    return acc, report, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--holdout_signers", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    files = find_seq_files(args.data_dir)
    print("Found sequence files:", len(files))
    train_files, val_files, test_files = split_files(files, args.holdout_signers)
    print("Train/Val/Test counts:", len(train_files), len(val_files), len(test_files))

    # build dataset to get label map
    ds_full = SeqDataset(train_files + val_files + test_files)
    labels = sorted(set([Path(f).parent.name for f in files]))
    label_to_idx = {l:i for i,l in enumerate(labels)}
    idx_to_label = {i:l for l,i in label_to_idx.items()}

    # datasets & loaders
    train_ds = SeqDataset(train_files); train_ds.label_to_idx = label_to_idx
    val_ds = SeqDataset(val_files); val_ds.label_to_idx = label_to_idx
    test_ds = SeqDataset(test_files); test_ds.label_to_idx = label_to_idx

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    # infer dims from first file
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
    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss=0; tot=0; correct=0
        for X, M, y, pths, metas in train_loader:
            X=X.to(dev); M=M.to(dev); y=y.to(dev)
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
        if val_acc > best_val:
            best_val = val_acc
            Path(args.out).mkdir(parents=True, exist_ok=True)
            ck = {"model_state": model.state_dict(), "labels": labels}
            torch.save(ck, Path(args.out)/"best.pt")
            print("Saved best model.")

    print("Training done. Best val acc:", best_val)
    # final evaluate on test if exists
    if len(test_files) > 0:
        ck = torch.load(Path(args.out)/"best.pt", map_location=dev)
        model.load_state_dict(ck["model_state"])
        test_acc, test_rep, test_cm = evaluate(model, test_loader, dev, idx_to_label)
        print("Test acc:", test_acc)
        print("Test classification report:")
        print(test_rep)
        # save final model
        torch.save({"model_state": model.state_dict(), "labels": labels}, Path(args.out)/"final_model.pt")
        print("Saved final model.")

if __name__ == "__main__":
    main()
