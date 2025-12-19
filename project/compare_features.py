#!/usr/bin/env python3
"""
compare_features.py

Usage:
  python compare_features.py --log1 log_script.json --log2 log_backend_2025-11-10T09-12-34.json

Expects each JSON to have keys:
  - feats_arr : list of [D] vectors (shape T x D)
  - mask      : list of ints (0/1) length T

Produces:
  - prints summary statistics and top-diff feature indices
  - saves PNG histograms for top differing feature indices (in out_dir)
"""
import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt

def load_json_log(path):
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    feats = np.array(js.get("feats_arr", []), dtype=np.float32)
    mask = np.array(js.get("mask", []), dtype=np.uint8)
    meta = js.get("meta", {})
    return feats, mask, meta

def pad_or_trim(a, target_len):
    if a.shape[0] == target_len:
        return a
    if a.shape[0] > target_len:
        return a[:target_len]
    # pad with zeros
    pad = np.zeros((target_len - a.shape[0], a.shape[1]), dtype=a.dtype)
    return np.vstack([a, pad])

def align_and_compare(feats1, feats2):
    # ensure same D
    if feats1.shape[1] != feats2.shape[1]:
        raise ValueError("Feature dimension mismatch: %d vs %d" % (feats1.shape[1], feats2.shape[1]))
    # align length by trimming to min length
    minT = min(feats1.shape[0], feats2.shape[0])
    f1 = feats1[:minT]
    f2 = feats2[:minT]
    return f1, f2

def summary_stats(feats):
    return feats.mean(axis=0), feats.std(axis=0)

def top_diffs(mean1, mean2, std1, std2, top_n=10):
    dmean = np.abs(mean1 - mean2)
    dstd  = np.abs(std1 - std2)
    mean_idx = np.argsort(-dmean)[:top_n]
    std_idx  = np.argsort(-dstd)[:top_n]
    return mean_idx, dmean[mean_idx], std_idx, dstd[std_idx]

def plot_hist(feats_a, feats_b, idx, out_path, label_a="log1", label_b="log2"):
    plt.figure(figsize=(7,4))
    plt.hist(feats_a[:, idx], bins=60, alpha=0.5, label=label_a)
    plt.hist(feats_b[:, idx], bins=60, alpha=0.5, label=label_b)
    plt.title(f"Feature index {idx} distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log1', required=True, help='Path to script log JSON (feats_arr + mask)')
    ap.add_argument('--log2', required=True, help='Path to backend log JSON (feats_arr + mask)')
    ap.add_argument('--top_n', type=int, default=8)
    ap.add_argument('--out_dir', default='compare_out')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    feats1, mask1, meta1 = load_json_log(args.log1)
    feats2, mask2, meta2 = load_json_log(args.log2)

    if feats1.size == 0 or feats2.size == 0:
        print("One of the logs has empty feats_arr. Exiting.")
        return

    if feats1.shape[1] != feats2.shape[1]:
        print("Feature dimension mismatch:", feats1.shape, feats2.shape)
        # try to trim to min columns
        minD = min(feats1.shape[1], feats2.shape[1])
        feats1 = feats1[:, :minD]
        feats2 = feats2[:, :minD]
        print("Trimmed both to D =", minD)

    # Align by min time length
    f1, f2 = align_and_compare(feats1, feats2)
    print("Aligned shapes:", f1.shape, f2.shape)
    print("Mask sums (valid frames):", int(mask1.sum()) if mask1.size else "N/A", "/", mask1.size if mask1.size else "N/A",
          " vs ", int(mask2.sum()) if mask2.size else "N/A", "/", mask2.size if mask2.size else "N/A")

    mean1, std1 = summary_stats(f1)
    mean2, std2 = summary_stats(f2)

    mean_idx, dmeans, std_idx, dstds = top_diffs(mean1, mean2, std1, std2, args.top_n)

    print("\nTop feature mean differences:")
    for i, d in zip(mean_idx, dmeans):
        print(f"  idx {i}: mean1={mean1[i]:.6f}, mean2={mean2[i]:.6f}, |diff|={d:.6f}")

    print("\nTop feature std differences:")
    for i, d in zip(std_idx, dstds):
        print(f"  idx {i}: std1={std1[i]:.6f}, std2={std2[i]:.6f}, |diff|={d:.6f}")

    # Save histograms for the top indices (mean diffs)
    for j, idx in enumerate(mean_idx[:min(len(mean_idx), 6)]):
        outp = os.path.join(args.out_dir, f"hist_mean_diff_idx{idx}.png")
        plot_hist(f1, f2, idx, outp, label_a="script", label_b="backend")
        print("Saved histogram:", outp)

    # Also print a small CSV of means/stds for manual inspection
    csvpath = os.path.join(args.out_dir, "feature_stats_summary.csv")
    with open(csvpath, "w", encoding="utf-8") as fo:
        fo.write("idx,mean_script,std_script,mean_backend,std_backend,diff_mean,diff_std\n")
        for i in range(len(mean1)):
            fo.write(f"{i},{mean1[i]:.6f},{std1[i]:.6f},{mean2[i]:.6f},{std2[i]:.6f},{abs(mean1[i]-mean2[i]):.6f},{abs(std1[i]-std2[i]):.6f}\n")
    print("Saved CSV summary:", csvpath)

if __name__ == "__main__":
    main()
