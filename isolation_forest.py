"""
isolation_forest.py — Isolation Forest baseline on SKAB
========================================================
Runner script for the Isolation Forest anomaly detector.
Shares the train/test split protocol with oc_qsvm_skab.py.

Usage:
    python isolation_forest.py                      # all files, seed 0
    python isolation_forest.py --files 5 6 --seeds 0 1
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Shared data utilities from the main benchmark script
from oc_qsvm_skab import (
    load_skab_file,
    get_features,
    build_train_test,
    _metrics,
)
from core.Isolation_Forest import Isolation_Forest


# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_file(path: str, args, seed: int) -> dict:
    fname = os.path.basename(path)

    df    = load_skab_file(path)
    feats = get_features(df)

    X_train_raw, X_test_raw, y_test, split_info = build_train_test(
        df, feats, args.n_train, args.n_test
    )

    # Standardise (Isolation Forest is not scale-invariant in practice)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    # Contamination = empirical anomaly ratio of the test window, clipped to
    # the valid [0.01, 0.50] range required by sklearn.
    contamination = float(np.clip(y_test.mean(), 0.01, 0.50))

    model = Isolation_Forest([seed, -1, contamination])
    model.fit(X_train)

    # predict() returns +1 (normal) or -1 (anomaly)
    raw_pred = model.predict(X_test)
    y_pred   = (raw_pred == -1).astype(int)
    m        = _metrics(y_test, y_pred)

    return {
        "file":           fname,
        "seed":           seed,
        "n_train":        len(X_train),
        "n_test":         len(y_test),
        "n_anomaly_test": int(y_test.sum()),
        "anomaly_ratio":  round(float(y_test.mean()), 4),
        "file_ratio":     split_info["file_ratio"],
        "fault_start":    split_info["fault_start"],
        "contamination":  round(contamination, 4),
        **{f"if_{k}": v for k, v in m.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list) -> None:
    df  = pd.DataFrame(all_results)
    sep = "─" * 60

    print(f"\n{sep}")
    print(" SUMMARY — Isolation Forest  (mean ± std across files and seeds)")
    print(f" n_runs = {len(df)}"
          f"  (files × seeds = {df['file'].nunique()} × {df['seed'].nunique()})")
    print(sep)

    for metric, col in [("F1", "if_f1"), ("FAR", "if_far"), ("MAR", "if_mar")]:
        vals = df[col].dropna()
        print(f"  {metric:<6} {vals.mean():.4f} ± {vals.std():.4f}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Isolation Forest baseline on SKAB"
    )
    p.add_argument("--data_dir", default=os.path.join("data", "skab", "others"))
    p.add_argument("--files",    nargs="+", type=int, default=list(range(5, 15)))
    p.add_argument("--n_train",  type=int, default=240,
                   help="Training samples (default: 240, all normal)")
    p.add_argument("--n_test",   type=int, default=60,
                   help="Total test samples (default: 60)")
    p.add_argument("--seeds",    nargs="+", type=int, default=[0])
    p.add_argument("--out_csv",  default=os.path.join("results", "if_results.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    paths = [
        os.path.join(args.data_dir, f"{n}.csv")
        for n in args.files
        if os.path.exists(os.path.join(args.data_dir, f"{n}.csv"))
    ]
    missing = [
        n for n in args.files
        if not os.path.exists(os.path.join(args.data_dir, f"{n}.csv"))
    ]
    if missing:
        print(f"[WARNING] Files not found: {missing}")
        print("          Run first: python download_skab.py\n")
    if not paths:
        sys.exit("[ERROR] No data files available.")

    all_results = []
    total = len(paths) * len(args.seeds)
    done  = 0

    for seed in args.seeds:
        for path in paths:
            done += 1
            fname = os.path.basename(path)
            print(f"[{done:>3}/{total}] {fname}  seed={seed}")
            try:
                r = run_file(path, args, seed)
                all_results.append(r)
                print(
                    f"         IF     "
                    f"F1={r['if_f1']:.4f}  "
                    f"FAR={r['if_far']:.4f}  "
                    f"MAR={r['if_mar']:.4f}"
                )
            except Exception as e:
                print(f"  [ERROR] {e}")

    if not all_results:
        sys.exit("[ERROR] No results produced.")

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")
    print_summary(all_results)


if __name__ == "__main__":
    main()
