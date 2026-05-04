#!/bin/bash
# run_all.sh — Execute all 6 anomaly detection methods
# =====================================================
# Usage:
#   bash run_all.sh            → full run (all files, 1 seed)
#   bash run_all.sh --dry-run  → fast validation (1 file, 1 seed, 2 qubits, ~3 min)

set -e
mkdir -p results results/plots

DRY_RUN=0
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=1
fi

echo "=============================================="
echo " OC-QSVM BENCHMARK — SKAB others/5-14"
echo " $(date)"
if [[ $DRY_RUN -eq 1 ]]; then
    echo " DRY-RUN MODE: 1 file (5.csv), 1 seed, 2 qubits"
fi
echo "=============================================="

echo ""
echo "[1/5] OC-SVM RBF + OC-QSVM qIT"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "      Dry-run: n_qubits=2, n_shots=200, n_train=120, n_test=50"
    python oc_qsvm_skab.py \
        --files 5 \
        --seeds 0 \
        --n_qubits 2 \
        --n_shots 200 \
        --n_train 120 \
        --n_test 50 \
        --nu 0.1 \
        --methods rbf qit \
        --out_csv results/skab_dryrun.csv

    echo ""
    echo "=============================================="
    echo " DRY-RUN COMPLETE"
    echo "=============================================="
    python3 -c "
import pandas as pd
df = pd.read_csv('results/skab_dryrun.csv')
rbf = df['rbf_f1'].iloc[0]
qit = df['qit_f1'].iloc[0]
rbf_far = df['rbf_far'].iloc[0]
qit_far = df['qit_far'].iloc[0]
print(f'  RBF: F1={rbf:.4f}  FAR={rbf_far:.4f}')
print(f'  qIT: F1={qit:.4f}  FAR={qit_far:.4f}')
print()
# F1 values may be coincidentally similar; FAR/MAR difference confirms
# that the kernels produce genuinely distinct decision boundaries.
if abs(rbf_far - qit_far) >= 0.05 or abs(rbf - qit) > 0.01:
    print('  OK  Kernels produce distinct predictions.')
else:
    print('  WARNING  Predictions may still be identical — check preprocessing.')
"
    exit 0
fi

# Full run — paper configuration
echo "      n_train=240 | n_test=60 | n_qubits=4 | n_shots=1000 | seed=0"
python oc_qsvm_skab.py \
    --files 5 6 7 8 9 10 11 12 13 14 \
    --seeds 0 \
    --n_qubits 4 \
    --n_shots 1000 \
    --n_train 240 \
    --n_test 60 \
    --nu 0.1 \
    --methods rbf qit \
    --out_csv results/skab_results.csv
echo "      OK  RBF + qIT complete"

echo ""
echo "[2/5] Isolation Forest"
python isolation_forest.py
echo "      OK  Isolation Forest complete"

echo ""
echo "[3/5] Conv-AE"
python Conv_AE.py
echo "      OK  Conv-AE complete"

echo ""
echo "[4/5] LSTM-AE"
python LSTM_AE.py
echo "      OK  LSTM-AE complete"

echo ""
echo "[5/5] MSCRED"
python mscred.py
echo "      OK  MSCRED complete"

echo ""
echo "=============================================="
echo " ALL METHODS COMPLETE — $(date)"
echo " Results saved to: results/"
echo "=============================================="
ls -lh results/*.csv