"""
mscred.py — MSCRED baseline on SKAB
=====================================
Runner script for the Multi-Scale Convolutional Recurrent Encoder-Decoder.
Shares the train/test split protocol with oc_qsvm_skab.py.

Preprocessing — multi-scale signature matrices:
  For each pair of sensors (i, j) and each scale w in SCALES:
      S_w[t, i, j] = (x_i[t-w:t] · x_j[t-w:t]) / w
  This captures inter-sensor correlations at different temporal resolutions.

  MSCRED input  shape: (samples, step_max, sensor_n, sensor_n, scale_n)
  MSCRED target shape: (samples, sensor_n, sensor_n, scale_n)
  (target = the signature matrix of the last step in the sequence)

Anomaly scoring:
  Reconstruction MSE of the last-step signature matrix, thresholded at
  mean(train_err) + k·std(train_err).

Usage:
    python mscred.py                         # all files, seed 0
    python mscred.py --files 5 6 --seeds 0 1
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
from core.MSCRED import MSCRED

# Multi-scale window sizes (inner-product window lengths for signature matrices)
DEFAULT_SCALES   = [1, 2, 5]
# Number of temporal steps per input sequence
DEFAULT_STEP_MAX = 5
# Training epochs (MSCRED default in core/ is 25)
DEFAULT_EPOCHS   = 25
DEFAULT_BATCH    = 200


# ─────────────────────────────────────────────────────────────────────────────
# Signature matrix preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def compute_signature_matrix(X: np.ndarray, t: int, scale: int) -> np.ndarray:
    """
    Compute the (sensor_n × sensor_n) signature matrix at time t for a given scale.

    S[i, j] = (x_i[t-scale:t] · x_j[t-scale:t]) / scale

    Parameters
    ----------
    X     : (T, sensor_n) standardised sensor array
    t     : end index (exclusive) of the window — corresponds to 'now'
    scale : window length (inner-product span)

    Returns
    -------
    S : (sensor_n, sensor_n)
    """
    window = X[t - scale: t, :]   # (scale, sensor_n)
    return (window.T @ window) / scale


def build_dataset(
    X: np.ndarray,
    scales: list,
    step_max: int,
) -> tuple:
    """
    Build MSCRED input/target arrays from a standardised 2-D sensor array.

    For each valid end-point t in [max_scale + step_max - 1, T):
      • input  — sequence of step_max signature-matrix stacks ending at t:
                 shape (step_max, sensor_n, sensor_n, scale_n)
      • target — signature-matrix stack at the last step (t):
                 shape (sensor_n, sensor_n, scale_n)

    Parameters
    ----------
    X        : (T, sensor_n) standardised array
    scales   : list of scale values (e.g. [1, 2, 5])
    step_max : temporal depth of the input sequence

    Returns
    -------
    X_seq : (n_samples, step_max, sensor_n, sensor_n, scale_n)
    Y_seq : (n_samples, sensor_n, sensor_n, scale_n)
    """
    T        = X.shape[0]
    sensor_n = X.shape[1]
    scale_n  = len(scales)
    max_scale = max(scales)
    start    = max_scale + step_max - 1   # first valid end-point

    if T <= start:
        raise ValueError(
            f"Not enough data to build MSCRED sequences: "
            f"T={T}, need > {start} (max_scale={max_scale} + step_max={step_max} - 1)."
        )

    n_samples = T - start
    X_seq = np.zeros((n_samples, step_max, sensor_n, sensor_n, scale_n),
                     dtype=np.float32)
    Y_seq = np.zeros((n_samples, sensor_n, sensor_n, scale_n),
                     dtype=np.float32)

    for idx in range(n_samples):
        t_end = idx + start          # end index of the window (inclusive)
        for s_idx in range(step_max):
            # temporal step s_idx corresponds to time-point t_end - (step_max-1-s_idx)
            t_step = t_end - (step_max - 1 - s_idx)
            for sc_idx, sc in enumerate(scales):
                X_seq[idx, s_idx, :, :, sc_idx] = compute_signature_matrix(
                    X, t_step, sc
                )
        # Target = last step
        Y_seq[idx] = X_seq[idx, -1]

    return X_seq, Y_seq


# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_file(path: str, args, seed: int) -> dict:
    fname    = os.path.basename(path)
    scales   = args.scales
    step_max = args.step_max

    df    = load_skab_file(path)
    feats = get_features(df)
    sensor_n = len(feats)
    scale_n  = len(scales)

    X_train_raw, X_test_raw, y_test, split_info = build_train_test(
        df, feats, args.n_train, args.n_test
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    X_train_seq, Y_train_seq = build_dataset(X_train, scales, step_max)
    X_test_seq,  Y_test_seq  = build_dataset(X_test,  scales, step_max)

    # MSCRED constructor: [sensor_n, scale_n, step_max]
    model = MSCRED([sensor_n, scale_n, step_max])
    model.fit(
        X_train_seq, Y_train_seq,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Reconstruction errors — MSE over all (sensor_n × sensor_n × scale_n) elements
    Y_train_pred  = model.predict(X_train_seq)
    train_errors  = np.mean((Y_train_seq - Y_train_pred) ** 2, axis=(1, 2, 3))

    Y_test_pred   = model.predict(X_test_seq)
    test_errors_w = np.mean((Y_test_seq - Y_test_pred)  ** 2, axis=(1, 2, 3))

    # Map windowed scores back to per-sample indices in X_test.
    # Sequence index i ends at test sample t = i + max(scales) + step_max - 1.
    max_scale   = max(scales)
    offset      = max_scale + step_max - 1
    n_test      = len(X_test)
    test_scores = np.full(n_test, np.nan)
    for i, err in enumerate(test_errors_w):
        t = i + offset
        if t < n_test:
            test_scores[t] = err

    valid_mask = ~np.isnan(test_scores)
    if valid_mask.sum() == 0:
        raise ValueError("No valid test scores — test window too short for MSCRED.")

    threshold    = train_errors.mean() + args.threshold_k * train_errors.std()
    y_test_eval  = y_test[valid_mask]
    scores_eval  = test_scores[valid_mask]
    y_pred_eval  = (scores_eval > threshold).astype(int)
    m            = _metrics(y_test_eval, y_pred_eval)

    return {
        "file":           fname,
        "seed":           seed,
        "n_train":        len(X_train),
        "n_test":         len(y_test),
        "n_anomaly_test": int(y_test.sum()),
        "anomaly_ratio":  round(float(y_test.mean()), 4),
        "file_ratio":     split_info["file_ratio"],
        "fault_start":    split_info["fault_start"],
        "n_scored":       int(valid_mask.sum()),
        "threshold":      round(float(threshold), 6),
        **{f"mscred_{k}": v for k, v in m.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results: list) -> None:
    df  = pd.DataFrame(all_results)
    sep = "─" * 60

    print(f"\n{sep}")
    print(" SUMMARY — MSCRED  (mean ± std across files and seeds)")
    print(f" n_runs = {len(df)}"
          f"  (files × seeds = {df['file'].nunique()} × {df['seed'].nunique()})")
    print(sep)

    for metric, col in [
        ("F1",  "mscred_f1"),
        ("FAR", "mscred_far"),
        ("MAR", "mscred_mar"),
    ]:
        vals = df[col].dropna()
        print(f"  {metric:<6} {vals.mean():.4f} ± {vals.std():.4f}")

    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MSCRED baseline on SKAB"
    )
    p.add_argument("--data_dir",  default=os.path.join("data", "skab", "others"))
    p.add_argument("--files",     nargs="+", type=int, default=list(range(5, 15)))
    p.add_argument("--n_train",   type=int, default=240)
    p.add_argument("--n_test",    type=int, default=60)
    p.add_argument(
        "--scales", nargs="+", type=int, default=DEFAULT_SCALES,
        help=f"Multi-scale window sizes (default: {DEFAULT_SCALES}). "
             "Larger scales require more history in each time step.",
    )
    p.add_argument(
        "--step_max", type=int, default=DEFAULT_STEP_MAX,
        help=f"Temporal depth of each MSCRED input sequence (default: {DEFAULT_STEP_MAX}).",
    )
    p.add_argument(
        "--threshold_k", type=float, default=2.0,
        help="threshold = mean(train_err) + k * std(train_err)  (default: 2.0)",
    )
    p.add_argument("--epochs",     type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--seeds",      nargs="+", type=int, default=[0])
    p.add_argument(
        "--out_csv", default=os.path.join("results", "mscred_results.csv")
    )
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
                    f"         MSCRED  "
                    f"F1={r['mscred_f1']:.4f}  "
                    f"FAR={r['mscred_far']:.4f}  "
                    f"MAR={r['mscred_mar']:.4f}"
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
