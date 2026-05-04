# data/skab/others/

This directory contains the SKAB benchmark CSV files used in all experiments.

**Source:** [SKAB — Skoltech Anomaly Benchmark](https://github.com/waico/SKAB)  
**Citation:** Katser, I. & Kozitsin, V. (2020). Skoltech Anomaly Benchmark (SKAB).

---

## Files

| File |
|------|
| `5.csv` |
| `6.csv` |
| `7.csv` |
| `8.csv` |
| `9.csv` |
| `10.csv` |
| `11.csv` |
| `12.csv` |
| `13.csv` |
| `14.csv` |

---

## Format

Each file is a semicolon-separated CSV with a `datetime` index:

```
datetime;Accelerometer1RMS;Accelerometer2RMS;Current;Pressure;
Temperature;Thermocouple;Voltage;Volume Flow RateRMS;anomaly;changepoint
```

- **Sensor columns (8):** continuous multivariate time series from a hydraulic pump testbed.
- **`anomaly`:** binary label — 0 = normal, 1 = anomalous. Used as ground truth.
- **`changepoint`:** binary label for abrupt distribution shifts. Not used in this work.

---

## Download

```bash
python download_skab.py
```

This script clones the SKAB repository and copies the relevant files here.
Alternatively, download manually from the [SKAB repo](https://github.com/waico/SKAB)
under `other/`.

---

## Anomaly Statistics (approximate, varies per file)

The anomaly ratio varies across files, typically in the range of 20–45%.
The exact ratio per file is computed at runtime and used to determine the
normal/anomaly composition of the test window.