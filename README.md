# OC-QSVM Benchmark — SKAB Dataset

**Anomaly detection with OC-QSVM vs classical and deep learning baselines**

Reproduces and extends the experiments of:
- Kölle et al. (2023) — *One-Class Support Vector Machines with Quantum Kernels*, arXiv:2312.09174
- Kölle et al. (2024) — *Quantum Kernel Methods for Anomaly Detection*, arXiv:2407.20753

Implementation based on: [q-anomaly](https://github.com/AfraeA/q-anomaly)  
Dataset: [SKAB](https://github.com/waico/SKAB) — Skoltech Anomaly Benchmark (Katser & Kozitsin, 2020)

---

## Methods

| Method | Type | File |
|--------|------|------|
| OC-SVM RBF | Classical kernel | `oc_qsvm_skab.py` |
| OC-QSVM qIT | Quantum Inversion Test kernel | `oc_qsvm_skab.py` |
| OC-QSVM qRM | Quantum Randomized Measurements kernel | `oc_qsvm_skab.py` |
| Isolation Forest | Ensemble (tree-based) | `isolation_forest.py` |
| Conv-AE | Deep learning (CNN autoencoder) | `Conv_AE.py` |
| LSTM-AE | Deep learning (LSTM autoencoder) | `LSTM_AE.py` |
| MSCRED | Deep learning (CNN + LSTM) | `mscred.py` |

---

## Data Split — Unified Protocol

All methods share the identical train/test split to ensure a fair comparison:

```
Train : 240 consecutive normal samples immediately preceding the test window
        (anomaly-free by construction; no label leakage)

Test  : 60 samples centered on the first fault onset
        normal/anomaly composition = empirical file anomaly ratio
        window: [fault_start − n_norm : fault_start + n_anom]
```

**Files used:** `others/5.csv` through `others/14.csv`

This design guarantees:
- All methods are evaluated on the exact same 60 test points
- Training data is always temporally prior to the test window (no leakage)
- Temporal ordering is preserved for sequence-based models (AEs)

---

## Repository Structure

```
oc-qsvm-skab/
├── README.md
├── requirements.txt
├── run_all.sh                 ← runs all methods (--dry-run available)
│
├── oc_qsvm_skab.py            ← OC-SVM RBF + OC-QSVM qIT/qRM
├── isolation_forest.py        ← Isolation Forest baseline
├── Conv_AE.py                 ← Convolutional Autoencoder
├── LSTM_AE.py                 ← LSTM Autoencoder
├── mscred.py                  ← MSCRED
│
├── core/                      ← model classes from SKAB repo
│   ├── __init__.py
│   ├── Conv_AE.py
│   ├── LSTM_AE.py
│   ├── MSCRED.py
│   └── Isolation_Forest.py
│
├── Kernel_calculation.py      ← quantum kernel (q-anomaly repo, GPU-patched)
├── download_skab.py           ← downloads SKAB CSVs
│
├── data/skab/others/          ← SKAB CSV files (auto-created by download_skab.py)
│   └── 5.csv … 14.csv
├── InterimResults/            ← quantum kernel cache (auto-created)
│   └── qIT/{train,test}/
└── results/                   ← output CSVs (auto-created)
    ├── skab_results.csv
    ├── skab_isolation_forest.csv
    ├── skab_conv_ae.csv
    ├── skab_lstm_ae.csv
    └── skab_mscred.csv
```

---

## Installation

### 1. Clone dependencies

```bash
# SKAB dataset + core model classes
git clone https://github.com/waico/SKAB.git
cp -r SKAB/core ./core
touch core/__init__.py      # required for Python imports

# Quantum kernel implementation
git clone https://github.com/AfraeA/q-anomaly.git
cp q-anomaly/Kernel_calculation.py ./Kernel_calculation.py
```

### 2. Create environment

```bash
conda create -n qanomaly python=3.10
conda activate qanomaly
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. GPU support (optional)

> **Critical:** `qiskit-aer` and `qiskit-aer-gpu` must never coexist.
> Always remove both before installing either one.

```bash
pip uninstall qiskit-aer qiskit-aer-gpu -y
pip install "qiskit-aer-gpu==0.15.1"
```

Verify GPU availability:
```bash
python -c "
from qiskit_aer import AerSimulator
sim = AerSimulator(device='GPU')
print('GPU OK:', sim.available_devices())
"
```
Expected output: `GPU OK: ['GPU', 'CPU']`

GPU acceleration uses **custatevec** from NVIDIA cuQuantum, installed automatically
as a dependency of `qiskit-aer-gpu`. Deep learning models (Conv-AE, LSTM-AE, MSCRED)
use TensorFlow, which detects CUDA automatically.

### 5. Download dataset

```bash
python download_skab.py
```

---

## Quickstart

### Dry-run (~3 min, validates end-to-end pipeline)

```bash
rm -rf InterimResults/
bash run_all.sh --dry-run
```

Expected output:
```
  RBF: F1=0.7xxx  FAR=0.7xxx
  qIT: F1=0.7xxx  FAR=0.4xxx
  OK  Kernels produce distinct predictions.
```


### Full run

```bash
bash run_all.sh
```

Or run methods individually:

```bash
# OC-SVM RBF and OC-QSVM qIT
python oc_qsvm_skab.py \
    --files 5 6 7 8 9 10 11 12 13 14 \
    --seeds 0 1 2 3 4 \
    --n_qubits 4 \
    --n_shots 1000 \
    --n_train 240 \
    --n_test 60 \
    --nu 0.1 \
    --methods rbf qit \
    --out_csv results/skab_results.csv

python isolation_forest.py
python Conv_AE.py
python LSTM_AE.py
python mscred.py
```

---

## Implementation Notes and Fixes

### Preprocessing for SKAB (qIT)

Kölle et al. use a `× 0.1` scaling after PCA, calibrated for the CC Fraud dataset
where features V1–V28 are compact Kaggle PCA components. For SKAB, PCA components
span approximately [−4, 4]:

```
× 0.1  → encoding angles in [−0.4, 0.4] rad
          K_off-diag ≈ 0.89, std = 0.07 → near-uniform kernel
          → OC-SVM converges to the same boundary as RBF → degenerate

MinMax[−1, 1] → encoding angles in [−1, 1] rad
                K_off-diag ≈ 0.27, std = 0.22 → discriminative kernel
                → distinct boundary from RBF (confirmed by different FAR/MAR)
```

**Fix:** `preprocess()` applies `MinMaxScaler(feature_range=(−1, 1))` for qIT,
fitted on training data only.

### RBF gamma

Kölle et al. use `γ = 1/(N × Var(X))`. With N = 240 this gives γ ≈ 0.004,
making K_RBF ≈ 1 for all pairs (degenerate). Fixed to `gamma='scale'`
(sklearn default: `1 / (n_features × Var(X))`).

### Cache isolation (composite seed)

The original `Kernel_calculation.py` uses only `(n_train, seed, n_qubits, n_shots)`
as the cache key — no reference to the data file. All files with the same configuration
shared a single cache entry, producing identical qIT outputs across files.

**Fix:** `composite_seed = seed × 1000 + file_id` (e.g. seed=1, file=8 → 1008).
Each (file, seed) pair now has a unique cache entry.

### Kernel PSD regularization

With finite shot counts, the qIT kernel matrix may have small negative eigenvalues
(shot noise ≈ 1/√n_shots). This is intrinsic to finite-sample quantum state estimation,
not an implementation error.

**Fix:** Tikhonov regularization before fitting:
```
K_reg = (K + K^T)/2 + ε·I    where ε = |λ_min| + 1e-6
```
This guarantees PSD and is the standard approach in quantum kernel learning
(Theis et al., 2023). Eigenvalue clipping followed by `fill_diagonal(K, 1)` was
tested but re-introduces negative eigenvalues; Tikhonov is preferred.

---

## Known Issues

### `Simulation device "GPU" is not supported`

**Cause:** Both `qiskit-aer` and `qiskit-aer-gpu` are installed simultaneously.
The CPU version takes precedence and rejects `device='GPU'`.

**Fix:**
```bash
pip uninstall qiskit-aer qiskit-aer-gpu -y
pip install "qiskit-aer-gpu==0.15.1"
```

### Qiskit version conflicts

`qiskit-aer-gpu==0.15.1` requires `qiskit 1.x`:
```
qiskit 1.4.3     % confirmed working
```

Full recovery:
```bash
pip uninstall qiskit qiskit-aer qiskit-aer-gpu qiskit-terra qiskit-ibmq-provider -y
pip cache purge
pip install "qiskit==1.4.3" "qiskit-aer-gpu==0.15.1"
```


### `from core.X import X` fails

```bash
touch core/__init__.py
```

---

## Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| F1 | 2·TP / (2·TP + FP + FN) | Primary metric — harmonic mean of precision and recall |
| FAR | FP / (FP + TN) | False alarm rate on normal samples |
| MAR | FN / (FN + TP) | Missed anomaly rate |

Results reported as **mean ± std** across files and seeds.

---

## Cache System

Cache key format (qIT):
```
InterimResults/qIT/{train|test}/dsize_{(N1,N2)}_seed_{composite_seed}_n_pc_{n_qubits}_n_shots_{n_shots}.npy
```

Where `composite_seed = seed × 1000 + file_id`.

**When to delete the cache:**

| Situation | Delete? |
|-----------|---------|
| Changed `n_qubits`, `n_shots`, or `n_train` | Yes |
| Running new seeds or files | No (new keys are created automatically) |
| Resuming an interrupted run | No |
| Modified kernel code | Yes |

```bash
rm -rf InterimResults/
```

---

## References

```bibtex
@article{kolle2023ocsvm,
  title={One-Class Support Vector Machines with Quantum Kernels},
  author={K{\"o}lle, Michael and others},
  journal={arXiv preprint arXiv:2312.09174},
  year={2023}
}

@article{kolle2024quantum,
  title={Quantum Kernel Methods for Anomaly Detection},
  author={K{\"o}lle, Michael and others},
  journal={arXiv preprint arXiv:2407.20753},
  year={2024}
}

@dataset{skab2020,
  title={{SKAB} — Skoltech Anomaly Benchmark},
  author={Katser, Iurii and Kozitsin, Viacheslav},
  year={2020},
  url={https://github.com/waico/SKAB}
}

@article{elben2022randomized,
  title={The randomized measurement toolbox},
  author={Elben, Andreas and others},
  journal={Nature Reviews Physics},
  volume={5},
  pages={9--24},
  year={2023}
}
```
