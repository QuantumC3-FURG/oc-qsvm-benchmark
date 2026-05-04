"""
oc_qsvm_skab.py — One-Class SVM with Quantum Kernels on the SKAB Dataset
=========================================================================
Reproduces and extends the anomaly detection experiments of Kölle et al.
(arXiv:2312.09174, arXiv:2407.20753) on the Skoltech Anomaly Benchmark.

Methods implemented:
  (1) OC-SVM RBF  — classical RBF kernel baseline
  (2) OC-QSVM qIT — quantum Inversion Test kernel (Kölle et al.)
  (3) OC-QSVM qRM — quantum Randomized Measurements kernel (optional, --methods qrm)

Preprocessing pipeline (per method):
  RBF : StandardScaler → PCA(n_qubits)
  qIT : StandardScaler → PCA(n_qubits) → MinMaxScaler([-1, 1])
  qRM : StandardScaler → PCA(n_qubits) → StandardScaler → × 1/√M

Primary metric: F1-score (anomaly class = 1).
Auxiliary metrics: FAR (false alarm rate) and MAR (missed anomaly rate).

Data split — unified across all baselines:
  Train : n_train normal samples immediately preceding the test window
          (guaranteed anomaly-free; no leakage by construction)
  Test  : n_test samples centered on the fault onset transition
          normal/anomaly composition = empirical file ratio

Quick start (1 seed, 2 files, RBF + qIT only):
    python oc_qsvm_skab.py --files 5 6 --seeds 0

Full run (paper configuration):
    python oc_qsvm_skab.py \\
        --files 5 6 7 8 9 10 11 12 13 14 \\
        --seeds 0 1 2 3 4 \\
        --n_qubits 4 --n_shots 1000 --nu 0.1 \\
        --methods rbf qit

Expected directory layout:
    project/
    ├── Kernel_calculation.py    ← quantum kernel (Kölle et al. / q-anomaly repo)
    ├── oc_qsvm_skab.py
    ├── data/skab/others/
    │   └── 5.csv … 14.csv
    ├── InterimResults/          (kernel cache — auto-created)
    └── results/                 (output CSVs — auto-created)
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Quantum kernel import ──────────────────────────────────────────────────────
try:
    from Kernel_calculation import (
        get_kernel_matrix_qIT,
        get_kernel_matrix_qRM,
    )
    QUANTUM_AVAILABLE = True
    print("[OK] Kernel_calculation.py loaded.")
except ImportError as e:
    print(f"[WARNING] Kernel_calculation.py not found: {e}")
    print("         Only the RBF baseline will be available.")
    QUANTUM_AVAILABLE = False

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

# SKAB sensors
SKAB_FEATURES = [
    "Accelerometer1RMS", "Accelerometer2RMS",
    "Current", "Pressure", "Temperature",
    "Thermocouple", "Voltage", "Volume Flow RateRMS",
]

def load_skab_file(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)

def get_features(df: pd.DataFrame) -> list:
    """Return available numeric sensor channels, falling back to all numeric
    non-label columns if canonical names are absent."""
    available = [c for c in SKAB_FEATURES if c in df.columns]
    if not available:
        skip = {"anomaly", "changepoint"}
        available = [c for c in df.columns
                     if df[c].dtype != object and c.lower() not in skip]
    return available

def build_train_test(df: pd.DataFrame, feats: list,
                     n_train: int, n_test: int):
    """
    Construct train/test splits using the unified protocol shared by all
    baselines (Isolation Forest, Conv-AE, LSTM-AE, MSCRED).

    Split procedure:
      1. Compute the empirical anomaly ratio of the file.
      2. Partition n_test proportionally into n_norm_test and n_anom_test.
      3. Define the test window centered on the first fault onset:
             [fault_start − n_norm_test : fault_start + n_anom_test]
      4. Training window: n_train samples immediately before the test window.
         All training samples are normal by construction (no leakage).

    Parameters
    ----------
    n_train : number of training samples (default 240, all normal).
    n_test  : total test samples (default 60); composition = file ratio.
    """
    label_col = next((c for c in ["anomaly", "Anomaly"] if c in df.columns), None)
    if label_col is None:
        raise ValueError("Column 'anomaly' not found in file.")

    X_all = df[feats].values
    y_all = df[label_col].values.astype(int)

    anomaly_idx = np.where(y_all == 1)[0]
    if len(anomaly_idx) == 0:
        raise ValueError("File contains no anomalous samples.")

    file_ratio  = float(y_all.mean())
    n_anom_test = round(n_test * file_ratio)
    n_norm_test = n_test - n_anom_test

    if n_anom_test == 0:
        raise ValueError(
            f"Anomaly ratio too low ({file_ratio:.3f}) for n_test={n_test}: "
            f"n_anom_test=0. Increase n_test or choose a file with more faults."
        )
    if n_norm_test == 0:
        raise ValueError(
            f"Anomaly ratio too high ({file_ratio:.3f}): n_norm_test=0."
        )

    # Test window centered on the first fault onset
    fault_start = int(anomaly_idx[0])
    test_start  = fault_start - n_norm_test
    test_end    = test_start + n_test

    if test_start < 0:
        raise ValueError(
            f"Insufficient normal samples before fault onset: "
            f"fault_start={fault_start}, n_norm_test={n_norm_test}."
        )
    if test_end > len(df):
        raise ValueError(
            f"Test window exceeds file length "
            f"(test_end={test_end} > len={len(df)})."
        )

    # Training window: n_train samples immediately before the test window
    train_start = max(0, test_start - n_train)
    train_end   = test_start

    if (train_end - train_start) < n_train:
        raise ValueError(
            f"Insufficient normal samples for training: "
            f"available={train_end - train_start}, required={n_train}."
        )

    X_train = X_all[train_start:train_end]
    X_test  = X_all[test_start:test_end]
    y_test  = y_all[test_start:test_end]

    return X_train, X_test, y_test, {
        "file_ratio":  round(file_ratio, 4),
        "n_anom_test": int(n_anom_test),
        "n_norm_test": int(n_norm_test),
        "fault_start": fault_start,
        "test_start":  test_start,
        "train_start": train_start,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (5.2 of Kölle et al., adapted for SKAB)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X_train_raw, X_test_raw, n_qubits, method, seed):
    """
    Shared pipeline: StandardScaler → PCA(n_qubits) → method-specific scaling.

    RBF : No additional scaling. sklearn's gamma='scale' handles normalization.

    qIT : MinMaxScaler([-1, 1]) applied after PCA.
          Kölle et al. use × 0.1, calibrated for CC Fraud where V1–V28 are
          already compact Kaggle PCA components. For SKAB, PCA components span
          approximately [-4, 4]:
            × 0.1  → encoding angles in [-0.4, 0.4] rad → K_off-diag ≈ 0.89,
                      std = 0.07. Near-uniform kernel → degenerate OC-SVM →
                      predictions identical to RBF regardless of n_train.
            MinMax → encoding angles in [-1, 1] rad → K_off-diag ≈ 0.27,
                      std = 0.22. Discriminative kernel → different decision
                      boundary from RBF (verified by distinct FAR/MAR).
          Scaler is fitted on training data only (no leakage).

    qRM : Second StandardScaler followed by × 1/√M (Haug et al. protocol).

    All scalers are fitted on X_train and applied to X_test.
    """
    from sklearn.preprocessing import MinMaxScaler as _MMS

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train_raw)
    Xte = scaler.transform(X_test_raw)

    pca = PCA(n_components=n_qubits, random_state=seed)
    Xtr = pca.fit_transform(Xtr)
    Xte = pca.transform(Xte)

    if method == "qit":
        sc_angles = _MMS(feature_range=(-1, 1))
        Xtr = sc_angles.fit_transform(Xtr)
        Xte = sc_angles.transform(Xte)

    elif method == "qrm":
        sc2 = StandardScaler()
        Xtr = sc2.fit_transform(Xtr)
        Xte = sc2.transform(Xte)
        factor = np.sqrt(1.0 / n_qubits)
        Xtr = Xtr * factor
        Xte = Xte * factor

    return Xtr, Xte

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def fit_rbf(X_train, nu):
    """
    OC-SVM with RBF kernel.

    gamma='scale' = 1 / (n_features × Var(X)), the sklearn default.
    Kölle et al. use gamma = 1 / (N × Var(X)); with N = 240 this yields
    gamma ≈ 0.004 → K_RBF ≈ 1 for all pairs → degenerate SVM.
    Using n_features in the denominator produces a well-conditioned kernel.
    """
    clf = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    clf.fit(X_train)
    return clf

def regularize_kernel(K: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Enforce positive semi-definiteness via Tikhonov regularization:
        K_reg = (K + K^T) / 2 + ε·I,  where ε = |λ_min| + 1e-6.

    With finite shot counts, the qIT kernel matrix accumulates shot noise of
    order 1/√n_shots per entry, which can produce small negative eigenvalues.
    This is not an implementation error — it is intrinsic to finite-sample
    quantum state estimation.

    Tikhonov regularization shifts all eigenvalues by ε, guaranteeing λ_i ≥ 0.
    The diagonal becomes 1 + ε ≈ 1 (ε is small, typically ≈ 0.03 at 1000 shots).
    This is the standard approach in quantum kernel learning (Theis et al., 2023).

    Note: eigenvalue clipping followed by fill_diagonal(K, 1) re-introduces
    negative eigenvalues and is therefore not used here.
    """
    K_sym   = (K + K.T) / 2
    min_eig = np.linalg.eigvalsh(K_sym).min()

    if min_eig >= -tol:
        return K_sym

    eps   = abs(min_eig) + 1e-6
    K_reg = K_sym + eps * np.eye(K_sym.shape[0])
    print(f"    [regularize_kernel] λ_min={min_eig:.6f} → Tikhonov ε={eps:.6f} applied")
    return K_reg

def fit_qit(X_train, nu, seed, n_shots):
    """Compute qIT kernel matrix, regularize, and fit OC-SVM."""
    K_raw = get_kernel_matrix_qIT(X_train, X_train,
                                   seed=seed, kmethod="qIT", n_shots=n_shots)
    K = regularize_kernel(K_raw)
    clf = OneClassSVM(kernel="precomputed", nu=nu)
    clf.fit(K)
    return clf, K

def fit_qrm(X_train, nu, seed, n_shots, n_settings):
    """Compute qRM kernel matrix, regularize, and fit OC-SVM."""
    K_raw = get_kernel_matrix_qRM(X_train, X_train,
                                   seed=seed, n_shots=n_shots, n_settings=n_settings)
    K = regularize_kernel(K_raw)
    clf = OneClassSVM(kernel="precomputed", nu=nu)
    clf.fit(K)
    return clf, K

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred):
    """
    Compute F1, FAR, and MAR for the anomaly class (label = 1).

    F1  = 2·TP / (2·TP + FP + FN)  — harmonic mean of precision and recall
    FAR = FP / (FP + TN)            — false alarm rate on normal samples
    MAR = FN / (FN + TP)            — fraction of anomalies missed
    """
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    mar = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

    return {"f1": f1, "far": far, "mar": mar}

def eval_rbf(clf, X_test, y_test):
    y_pred = (clf.predict(X_test) == -1).astype(int)
    return _metrics(y_test, y_pred)

def eval_qkernel(clf, X_train, X_test, y_test,
                 seed, n_shots, kmethod, n_settings=30):
    """Compute the test kernel matrix and predict anomaly labels."""
    if kmethod == "qIT":
        K_test = get_kernel_matrix_qIT(X_test, X_train,
                                        seed=seed, kmethod="qIT", n_shots=n_shots)
    else:
        K_test = get_kernel_matrix_qRM(X_test, X_train,
                                        seed=seed, n_shots=n_shots,
                                        n_settings=n_settings)
    y_pred = (clf.predict(K_test) == -1).astype(int)
    return _metrics(y_test, y_pred)

# ─────────────────────────────────────────────────────────────────────────────
# Per-file pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_file(path, args, seed):
    fname = os.path.basename(path)

    df    = load_skab_file(path)
    feats = get_features(df)

    X_train_raw, X_test_raw, y_test, split_info = build_train_test(
        df, feats, args.n_train, args.n_test
    )

    results = {
        "file":           fname,
        "seed":           seed,
        "n_train":        len(X_train_raw),
        "n_test":         len(y_test),
        "n_anomaly_test": int(y_test.sum()),
        "anomaly_ratio":  round(float(y_test.mean()), 4),
        "file_ratio":     split_info["file_ratio"],
        "fault_start":    split_info["fault_start"],
        "test_start":     split_info["test_start"],
    }

    # Composite seed: unique cache key per (file × seed) pair.
    # Format: composite_seed = seed × 1000 + file_id
    # Example: seed=1, file=8 → composite_seed=1008.
    # Prevents cache collisions across files sharing the same seed.
    try:
        file_id = int(os.path.splitext(fname)[0])
    except ValueError:
        file_id = abs(hash(fname)) % 1000
    composite_seed = seed * 1000 + file_id

    # ── RBF ──────────────────────────────────────────────────────────────────
    if "rbf" in args.methods:
        Xtr, Xte = preprocess(X_train_raw, X_test_raw, args.n_qubits, "rbf", seed)
        clf = fit_rbf(Xtr, args.nu)
        m   = eval_rbf(clf, Xte, y_test)
        results.update({f"rbf_{k}": v for k, v in m.items()})

    # ── qIT ──────────────────────────────────────────────────────────────────
    if "qit" in args.methods:
        if not QUANTUM_AVAILABLE:
            print("  [SKIP qIT] — Kernel_calculation.py not available")
        else:
            Xtr, Xte = preprocess(X_train_raw, X_test_raw, args.n_qubits, "qit", seed)
            t0 = time.time()
            clf, _ = fit_qit(Xtr, args.nu, composite_seed, args.n_shots)
            m = eval_qkernel(clf, Xtr, Xte, y_test, composite_seed,
                              args.n_shots, "qIT")
            results.update({f"qit_{k}": v for k, v in m.items()})
            results["qit_time"] = round(time.time() - t0, 1)

    # ── qRM ──────────────────────────────────────────────────────────────────
    if "qrm" in args.methods:
        if not QUANTUM_AVAILABLE:
            print("  [SKIP qRM] — Kernel_calculation.py not available")
        else:
            Xtr, Xte = preprocess(X_train_raw, X_test_raw, args.n_qubits, "qrm", seed)
            t0 = time.time()
            clf, _ = fit_qrm(Xtr, args.nu, composite_seed,
                              args.n_shots, args.n_rm_settings)
            m = eval_qkernel(clf, Xtr, Xte, y_test, composite_seed,
                              args.n_shots, "qRM", args.n_rm_settings)
            results.update({f"qrm_{k}": v for k, v in m.items()})
            results["qrm_time"] = round(time.time() - t0, 1)

    return results

# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, methods):
    df  = pd.DataFrame(all_results)
    sep = "─" * 68

    mean_ratio = df["anomaly_ratio"].mean() if "anomaly_ratio" in df.columns else float("nan")

    print(f"\n{sep}")
    print(" SUMMARY — F1 / FAR / MAR  (mean ± std across files and seeds)")
    print(f" n_runs = {len(df)}  "
          f"(files × seeds = {df['file'].nunique()} × {df['seed'].nunique()})")
    print(f" Split: {df['n_train'].iloc[0]} train | "
          f"{df['n_test'].iloc[0]} test (mean anomaly ratio: {mean_ratio:.1%})")
    print(f" FAR = FP/(FP+TN) | MAR = FN/(FN+TP)")
    print(sep)
    print(f" {'Method':<10}  {'F1':>14}  {'FAR':>12}  {'MAR':>12}")
    print(sep)

    for m in methods:
        col = f"{m}_f1"
        if col not in df.columns:
            continue
        f1  = df[f"{m}_f1"].dropna()
        far = df[f"{m}_far"].dropna()
        mar = df[f"{m}_mar"].dropna()
        print(
            f" {m.upper():<10}  "
            f"{f1.mean():.4f} ± {f1.std():.4f}  "
            f"{far.mean():.4f} ± {far.std():.4f}  "
            f"{mar.mean():.4f} ± {mar.std():.4f}"
        )

    print(sep)
    print()
    print(" Reference — Kölle et al. (CC Fraud, 6 features, n=500, 15 seeds):")
    print(f" {'RBF':<10}  F1 ≈ 0.30–0.50")
    print(f" {'qIT':<10}  F1 ≈ 0.30–0.50  (higher variance than RBF)")
    print(f" {'qRM unm.':<10}  F1 ≈ 0.20–0.40  (high variance)")
    print()
    print(" Validation criteria:")
    print("   • RBF and qIT should have comparable F1 (central result of the paper)")
    print("   • FAR and MAR must differ by ≥ 0.05 to confirm distinct decision boundaries")
    print("   • SKAB differs from CC Fraud — absolute values will differ, but")
    print("     relative ordering and qualitative behaviour should match.")
    print(sep)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="OC-QSVM on SKAB — F1-score as primary metric"
    )
    p.add_argument("--data_dir",      default=os.path.join("data", "skab", "others"))
    p.add_argument("--files",         nargs="+", type=int, default=list(range(5, 15)))
    p.add_argument("--n_train",       type=int,   default=240,
                   help="Training samples (default: 240) — all normal by construction")
    p.add_argument("--n_test",        type=int,   default=60,
                   help="Total test samples (default: 60); "
                        "normal/anomaly split = empirical file ratio")
    p.add_argument("--n_qubits",      type=int,   default=4)
    p.add_argument("--n_shots",       type=int,   default=1000,
                   help="Measurement shots per kernel element (paper: 1000 for qIT)")
    p.add_argument("--n_rm_settings", type=int,   default=30,
                   help="Number of random measurement settings (paper: 30 for qRM)")
    p.add_argument("--nu",            type=float, default=0.1,
                   help="OC-SVM nu parameter (paper: 0.1)")
    p.add_argument("--seeds",         nargs="+",  type=int, default=[0],
                   help="Random seeds (paper uses 0–14; start with [0] for testing)")
    p.add_argument("--methods",       nargs="+",
                   choices=["rbf", "qit", "qrm"],
                   default=["rbf", "qit"])
    p.add_argument("--out_csv",       default=os.path.join("results", "skab_results.csv"))
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    print(f"[Config] methods={args.methods} | qubits={args.n_qubits} | "
          f"nu={args.nu} | shots={args.n_shots} | seeds={args.seeds}")
    print(f"[Split]  train={args.n_train} normal samples | "
          f"test={args.n_test} total (ratio = empirical file ratio)\n")

    paths = [os.path.join(args.data_dir, f"{n}.csv") for n in args.files
             if os.path.exists(os.path.join(args.data_dir, f"{n}.csv"))]

    missing = [n for n in args.files
               if not os.path.exists(os.path.join(args.data_dir, f"{n}.csv"))]
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
                ratio_str = (f"ratio={r['anomaly_ratio']:.1%} "
                             f"({r['n_anomaly_test']}A/{r['n_test']}T)")
                print(f"         split  {ratio_str}")
                for m in args.methods:
                    f1  = r.get(f"{m}_f1",  float("nan"))
                    far = r.get(f"{m}_far", float("nan"))
                    mar = r.get(f"{m}_mar", float("nan"))
                    t   = r.get(f"{m}_time", "")
                    t_str = f"  {t}s" if t else ""
                    print(f"         {m.upper():<6} F1={f1:.4f}  FAR={far:.4f}  MAR={mar:.4f}{t_str}")
            except Exception as e:
                print(f"  [ERROR] {e}")

    if not all_results:
        sys.exit("[ERROR] No results produced.")

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(args.out_csv, index=False)
    print(f"\n[Saved] {args.out_csv}")

    print_summary(all_results, args.methods)


if __name__ == "__main__":
    main()