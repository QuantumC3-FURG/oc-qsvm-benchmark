# core/

This directory contains the model class definitions used across the benchmark.
Each file defines a single class and no benchmark logic is included here.
Runner scripts at the project root import from this package.

---

## Contents

| File | Model | Class |
|------|-------|-------|
| `__init__.py` | — | — |
| `Conv_AE.py` | Convolutional Autoencoder | `Conv_AE` |
| `LSTM_AE.py` | LSTM Autoencoder | `LSTM_AE` |
| `MSCRED.py` | Multi-Scale Convolutional Recurrent Encoder-Decoder | `MSCRED` |
| `Isolation_Forest.py` | Isolation Forest (scikit-learn wrapper) | `Isolation_Forest` |

---

## Architecture notes

- `Conv_AE`: Two `Conv1D(strides=2)` encoder layers and two `Conv1DTranspose(strides=2)`
  decoder layers, followed by a final `Conv1DTranspose` that restores the feature dimension.
  Input timesteps must be divisible by 4.
- `LSTM_AE`: LSTM encoder → RepeatVector → LSTM decoder → TimeDistributed Dense.
  The constructor accepts `[epochs, batch_size, val_split]`.
- `MSCRED`: Multi-scale ConvLSTM encoder with attention-weighted temporal aggregation
  and a convolutional decoder. The constructor accepts `[sensor_n, scale_n, step_max]`.
  `predict()` returns the reconstructed signature matrix (positive values).
- `Isolation_Forest`: Wrapper around `sklearn.ensemble.IsolationForest`.
  The constructor accepts `[random_state, n_jobs, contamination]`.
  `predict()` returns +1 for normal samples and -1 for anomalies.

## Dependencies

All deep learning models require TensorFlow >= 2.12. See `requirements.txt`.
