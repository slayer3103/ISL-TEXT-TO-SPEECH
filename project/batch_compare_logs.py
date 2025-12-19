#!/usr/bin/env python3
"""
batch_compare_logs.py

Batch-compare many log files from script vs backend.

Usage examples:

# automatic pairing by timestamp (default):
python batch_compare_logs.py --script_dir ./ --backend_dir ./static/debug_logs --out_dir compare_out --tol_secs 10

# manual pairing via CSV:
python batch_compare_logs.py --pairs manual_pairs.csv --out_dir compare_out

Output:
 - compare_out/batch_report.csv    (one line per pair with top diffs summary)
 - compare_out/<pairname>_stats.csv
 - compare_out/<pairname>_hist_idxNN.png
"""
import argparse, os, json, math, csv, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

ISO_TS_RE = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})')

def parse_ts_from_name(name):
    m = ISO_TS_RE.search(name)
    if not m:
        return None
    s = m.group(1)
    # parse with hyphen between time parts: YYYY-MM-DDTHH-MM-SS
    return datetime.strptime(s, "%Y-%m-%dT%H-%M-%S")

def load_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        j = json.load(f)
    feats = np.array(j.get("feats_arr", []), dtype=np.float32)
    mask = np.array(j.get("mask", []), dtype=np.uint8)
    meta = j.get("meta", {})
    return feats, mask, meta

def align_and_trim(f1, f2):
    # trim to min time steps
    minT = min(f1.shape[0], f2.shape[0])
    return f1[:minT], f2[:minT]

def summary_stats(feats):
    return feats.mean(axis=0), feats.std(axis=0)

def top_diffs(mean1, mean2, std1, std2, top_n=5):
    dmean = np.abs(mean1-mean2)
    dstd = np.abs(std1-std2)
    mi = np.argsort(-dmean)[:top_n]
    si = np.argsort(-dstd)[:top_n]
    return mi, dmean[mi], si, dstd[si]

def plot_hist(f1, f2, idx, out_path, lab1='script', lab2='backend'):
    plt.figure(figsize=(6,3.5))
    plt.hist(f1[:, idx], bins=60, alpha=0.5, label=lab1)
    plt.hist(f2[:, idx], bins=60, alpha=0.5, label=lab2)
    plt.title(f"feature idx {idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def pair_by_time(script_files, backend_files, tol_secs=10):
    # script_files, backend_files: list of Paths
    backend_ts = [(f, parse_ts_from_name(f.name)) for f in backend_files]
    pairs = []
    for s in script_files:
        st = parse_ts_from_name(s.name)
        if st is None:
            continue
        # find closest backend
        best = None; best_dt = None
        for b, bt in backend_ts:
            if bt is None: continue
            dt = abs((st - bt).total_seconds())
            if best is None or dt < best_dt:
                best = b; best_dt = dt
        if best is not None and best_dt is not None and best_dt <= tol_secs:
            pairs.append((s, best, best_dt))
    return pairs

def read_pairs_csv(path):
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row: continue
            a = Path(row[0].strip()); b = Path(row[1].strip())
            pairs.append((a,b, None))
    return pairs

def make_pairname(a,b):
    return f"{a.stem}__{b.stem}"

def compare_pair(a_path, b_path, out_dir, top_n=6):
    feats1, mask1, _ = load_log(a_path)
    feats2, mask2, _ = load_log(b_path)
    if feats1.size == 0 or feats2.size == 0:
        return {"pair": f"{a_path.name}__{b_path.name}", "error": "empty_feats"}

    if feats1.shape[1] != feats2.shape[1]:
        # trim columns to smallest
        D = min(feats1.shape[1], feats2.shape[1])
        feats1 = feats1[:, :D]; feats2 = feats2[:, :D]

    f1, f2 = align_and_trim(feats1, feats2)
    mean1, std1 = summary_stats(f1)
    mean2, std2 = summary_stats(f2)
    mi, dme, si, dst = top_diffs(mean1, mean2, std1, std2, top_n=top_n)

    pairname = make_pairname(a_path, b_path)
    out_csv = out_dir / f"{pairname}_stats.csv"
    with open(out_csv, 'w', encoding='utf-8') as fo:
        fo.write("idx,mean_script,std_script,mean_backend,std_backend,diff_mean,diff_std\n")
        for i in range(len(mean1)):
            fo.write(f"{i},{mean1[i]:.6f},{std1[i]:.6f},{mean2[i]:.6f},{std2[i]:.6f},{abs(mean1[i]-mean2[i]):.6f},{abs(std1[i]-std2[i]):.6f}\n")

    # save histograms for top mean diffs
    hist_paths = []
    for idx in mi[:6]:
        out_hist = out_dir / f"{pairname}_hist_idx{idx}.png"
        plot_hist(f1, f2, idx, out_hist, lab1=a_path.stem, lab2=b_path.stem)
        hist_paths.append(out_hist.name)

    summary = {
        "pair": f"{a_path.name}__{b_path.name}",
        "script": str(a_path),
        "backend": str(b_path),
        "rows_compared": int(min(f1.shape[0], f2.shape[0])),
        "top_mean_diff_idxs": [int(x) for x in mi.tolist()],
        "top_mean_diff_vals": [float(x) for x in dme.tolist()],
        "top_std_diff_idxs": [int(x) for x in si.tolist()],
        "top_std_diff_vals": [float(x) for x in dst.tolist()],
        "stats_csv": out_csv.name,
        "histograms": hist_paths
    }
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--script_dir', help='dir with script logs (or pattern)', default='.')
    ap.add_argument('--backend_dir', help='dir with backend logs', default='static/debug_logs')
    ap.add_argument('--pairs', help='optional CSV with manual pairs (script,backend) paths', default=None)
    ap.add_argument('--out_dir', help='output folder', default='compare_out')
    ap.add_argument('--tol_secs', type=float, default=10.0, help='max time diff to pair by timestamp')
    ap.add_argument('--top_n', type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs:
        pairs = read_pairs_csv(Path(args.pairs))
    else:
        script_files = sorted([p for p in Path(args.script_dir).glob("*.json") if p.name.startswith("log_script")])
        backend_files = sorted([p for p in Path(args.backend_dir).glob("*.json") if p.name.startswith("log_backend")])
        pairs = pair_by_time(script_files, backend_files, tol_secs=args.tol_secs)

    if not pairs:
        print("No pairs found. Check directories or provide manual pairs CSV.")
        return

    batch = []
    for a,b,dt in pairs:
        print("Comparing:", a.name, "<-->", b.name, " dt=", dt)
        try:
            res = compare_pair(a, b, out_dir, top_n=args.top_n)
            batch.append(res)
        except Exception as e:
            print("Failed compare for pair", a, b, e)

    # write batch_report.csv
    rpt = out_dir / "batch_report.csv"
    with open(rpt, 'w', encoding='utf-8') as fo:
        hdr = "pair,script,backend,rows_compared,top_mean_diff_idxs,top_mean_diff_vals,top_std_diff_idxs,top_std_diff_vals,stats_csv,histograms\n"
        fo.write(hdr)
        for item in batch:
            fo.write(f"{item['pair']},{item['script']},{item['backend']},{item['rows_compared']},"
                     f"\"{item['top_mean_diff_idxs']}\",\"{item['top_mean_diff_vals']}\","
                     f"\"{item['top_std_diff_idxs']}\",\"{item['top_std_diff_vals']}\","
                     f"{item['stats_csv']},\"{item['histograms']}\"\n")
    print("Wrote batch report to", rpt)

if __name__ == '__main__':
    main()
