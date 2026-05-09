"""
Conv_AE.py — Convolutional Autoencoder baseline on SKAB
=========================================================
Runner script for the Conv-AE anomaly detector.
Shares the train/test split protocol with oc_qsvm_skab.py.

Architecture constraint (from core/Conv_AE.py):
  Two Conv1D(strides=2) + two Conv1DTranspose(strides=2)
  → window_size must be divisible by 4.

Anomaly scoring:
  1. Train reconstruction errors   → threshold = mean + k·std
  2. Per-sample test error         → anomaly if error > threshold
  Window-to-sample mapping uses the mean of all windows that cover
  each sample (overlap-and-average).

Usage:
    python Conv_AE.py                         # all files, seed 0
    python Conv_AE.py --files 5 6 --seeds 0 1
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from oc_qsvm_skab import (
    load_skab_file,
    get_features,
    build_train_test,
    _metrics,
)
from core.Conv_AE import Conv_AE

# Default sliding-window size — must be divisible by 4
DEFAULT_WINDOW = 32


# ─────────────────────────────────────────────────────────────────────────────
# Windowing helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_windows(X: np.ndarray, window_size: int) -> np.ndarray:
    """
    Sliding-window segmentation over a 2-D array.

    Parameters
    ----------
    X           : (T, n_features)
    window_size : length of each window

    Returns
    -------
    windows : (T - window_size + 1, window_size, n_features)
    """
    n = len(X)
    if n < window_size:
        raise ValueError(
            f"Sequence length ({n}) is shorter than window_size ({window_size})."
        )
    return np.stack([X[i: i + window_size] for i in range(n - window_size + 1)])


def reconstruction_errors(model: Conv_AE, X_windows: np.ndarray) -> np.ndarray:
    """Per-window MSE between input and reconstruction."""
    X_pred = model.predict(X_windows)
    return np.mean((X_windows - X_pred) ** 2, axis=(1, 2))


def window_to_sample_scores(
    window_scores: np.ndarray, n_samples: int, window_size: int
) -> np.ndarray:
    """
    Map per-window scores to per-sample scores via overlap-and-average.

    Window i covers samples [i, i + window_size).
    The score for sample t is the mean of all window scores that contain t.
    """
    sample_scores = np.zeros(n_samples)
    counts        = np.zeros(n_samples)
    for i, s in enumerate(window_scores):
        lo = i
        hi = min(i + window_size, n_samples)
        sample_scores[lo:hi] += s
        counts[lo:hi]         += 1
    counts = np.maximum(counts, 1)          # avoid division by zero
    return sample_scores / counts


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

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    X_train_w = make_windows(X_train, args.window_size)
    X_test_w  = make_windows(X_test,  args.window_size)

    model = Conv_AE()
    model.fit(X_train_w)

    train_errors  = reconstruction_errors(model, X_train_w)
    test_errors_w = reconstruction_errors(model, X_test_w)

    test_scores = window_to_sample_scores(
        test_errors_w, len(X_test), args.window_size
    )

    # Threshold calibrated on training reconstruction errors
    threshold = train_errors.mean() + args.threshold_k * train_errors.std()
    y_pred    = (test_scores > threshold).astype(int)
    m         = _metrics(y_test, y_pred)

    return {
        "file":           fname,
        "seed":           seed,
        "n_train":        len(X_train),
        "n_test":         len(y_test),
        "n_anomaly_test": int(y_test.sum()),
        "anomaly_ratio":  round(float(y_test.mean()), 4),
        "file_ratio":     split_info["file_ratio"],
        "fault_start":    split_info["fault_start"],
        "window_size":    args.window_size,
        "threshold":      round(float(threshold), 6),
        **{f"cae_{k}": v for k, v in m.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list) -> None:
    df  = pd.DataFrame(all_results)
    sep = "─" * 60

    print(f"\n{sep}")
    print(" SUMMARY — Conv-AE  (mean ± std across files and seeds)")
    print(f" n_runs = {len(df)}"
          f"  (files × seeds = {df['file'].nunique()} × {df['seed'].nunique()})")
    print(sep)

    for metric, col in [("F1", "cae_f1"), ("FAR", "cae_far"), ("MAR", "cae_mar")]:
        vals = df[col].dropna()
        print(f"  {metric:<6} {vals.mean():.4f} ± {vals.std():.4f}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convolutional Autoencoder baseline on SKAB"
    )
    p.add_argument("--data_dir",    default=os.path.join("data", "skab", "others"))
    p.add_argument("--files",       nargs="+", type=int, default=list(range(5, 15)))
    p.add_argument("--n_train",     type=int, default=240)
    p.add_argument("--n_test",      type=int, default=60)
    p.add_argument(
        "--window_size", type=int, default=DEFAULT_WINDOW,
        help=f"Sliding-window length (default: {DEFAULT_WINDOW}); "
             "must be divisible by 4 due to Conv1D stride=2 constraint.",
    )
    p.add_argument(
        "--threshold_k", type=float, default=2.0,
        help="threshold = mean(train_err) + k * std(train_err)  (default: 2.0)",
    )
    p.add_argument("--seeds",   nargs="+", type=int, default=[0])
    p.add_argument(
        "--out_csv", default=os.path.join("results", "conv_ae_results.csv")
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.window_size % 4 != 0:
        sys.exit(
            f"[ERROR] window_size must be divisible by 4 "
            f"(got {args.window_size}). "
            "Conv1D(strides=2) × 2 requires this for symmetric reconstruction."
        )

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
                    f"         CAE    "
                    f"F1={r['cae_f1']:.4f}  "
                    f"FAR={r['cae_far']:.4f}  "
                    f"MAR={r['cae_mar']:.4f}"
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
