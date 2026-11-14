"""Collect Canny F1 and compute KID against real COCO validation images."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def kid(x, y, subset_size=1000, subsets=100, seed=0):
    """Unbiased KID between feature sets x (m x d) and y (n x d); returns (mean, std) over random subsets."""
    rng = np.random.default_rng(seed)
    s = min(subset_size, len(x), len(y))
    d = x.shape[1]
    vals = []
    for _ in range(subsets):
        a = x[rng.choice(len(x), s, replace=False)].astype(np.float64)
        b = y[rng.choice(len(y), s, replace=False)].astype(np.float64)
        kxx, kyy, kxy = (a @ a.T / d + 1) ** 3, (b @ b.T / d + 1) ** 3, (a @ b.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) + (kyy.sum() - np.trace(kyy))) / (s * (s - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--real_feats", type=Path, help="default: <out>/real/features.npy")
    ap.add_argument("--subset_size", type=int, default=1000)
    args = ap.parse_args()
    real_path = args.real_feats or args.out / "real" / "features.npy"
    if not real_path.is_file():
        ap.error(f"Missing real-image features: {real_path}; run evaluate first")
    if args.subset_size < 2:
        ap.error("--subset_size must be at least 2")
    real = np.load(real_path)
    if len(real) < 2:
        ap.error("KID requires at least two real-image features")

    rows = []
    for js in sorted(args.out.glob("*/step_*/*.json")):
        r = json.load(open(js))
        feats = np.load(js.with_suffix(".npy"))
        if len(feats) < 2:
            ap.error(f"KID requires at least two generated-image features: {js}")
        r["kid_real"], r["kid_real_std"] = kid(feats, real, args.subset_size)
        rows.append(r)
        print(f"{r['name']:18s} {r['step']:>6} {r['mode']:8s} F1 {r['f1']:.4f}  "
              f"KID real {r['kid_real']:.5f}")
    cols = ["name", "step", "mode", "n", "f1", "f1_sem", "precision", "recall",
            "kid_real", "kid_real_std"]
    if not rows:
        ap.error(f"No evaluation results in {args.out}")
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
