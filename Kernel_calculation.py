"""
Kernel_calculation.py — Quantum Kernel Computation for OC-QSVM
===============================================================
Source: q-anomaly repository (Kölle et al., arXiv:2312.09174).
Modifications in this file:
  - GPU-aware AerSimulator initialization with CPU fallback.
  - Composite seed support: cache keys include (file_id × seed) to prevent
    cross-file cache collisions when running multiple SKAB files.

Kernels implemented:
  qIT : Inversion Test kernel (probability of measuring the all-zeros state
        after applying U(x2)†·U(x1) to |0⟩).
  qRM : Randomized Measurements kernel (cross-purity estimator from
        local random Haar unitaries, Eq. 7 of arXiv:2108.01039).

Feature map: IQP-like circuit with 2×reuploads layers of H + Rz(x_i) + Rzz(x_i·x_j).
Reuploads = 3 (paper default).

Caching: Intermediate kernel matrices are stored as .npy files under
InterimResults/{kmethod}/{train|test}/ to allow resuming interrupted runs.
Cache key format for qIT:
    dsize_{(N1,N2)}_seed_{composite_seed}_n_pc_{n_qubits}_n_shots_{n_shots}.npy
"""

import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from itertools import product, chain

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator as QasmSimulator   # requires qiskit-aer >= 0.12 / Qiskit 1.x
from qiskit.quantum_info import random_unitary

ROOT_DIR = os.path.dirname(os.path.abspath("./__file__"))

# ── GPU availability check at import time ─────────────────────────────────────
def _test_gpu() -> bool:
    """
    Run a minimal 1-qubit circuit on AerSimulator(device='GPU').
    Returns True if GPU simulation succeeds, False otherwise.
    GPU is provided by qiskit-aer-gpu via custatevec (NVIDIA cuQuantum).
    """
    try:
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.measure_all()
        sim = QasmSimulator(device="GPU")
        from qiskit import transpile as _tp
        job = sim.run(_tp(qc, sim), shots=10)
        job.result()
        return True
    except Exception:
        return False

_USE_GPU = _test_gpu()
print(f"[Kernel] GPU: {'active' if _USE_GPU else 'unavailable — falling back to CPU'}")
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Feature map
# ─────────────────────────────────────────────────────────────────────────────

def quantum_feature(x, reuploads):
    """
    IQP-like data-encoding circuit with data re-uploading.

    Structure per re-upload layer (repeated 2×reuploads times):
      H on all qubits
      Rz(x_i) on qubit i          — single-qubit encoding
      Rzz(x_i · x_j) on (i, j)   — two-qubit interaction terms

    Paper uses reuploads=3 (6 layers total).
    """
    n_pc = len(x)
    qr = QuantumRegister(n_pc)
    cr = ClassicalRegister(n_pc)
    qc = QuantumCircuit(qr, cr)
    for r in range(2 * reuploads):
        qc.h(range(n_pc))
        qc.barrier()
        for i in range(n_pc):
            qc.rz(x[i], qr[i])
        qc.barrier()
        for i in range(n_pc):
            for j in range(i + 1, n_pc):
                qc.rzz(x[i] * x[j], i, j)
        qc.barrier()
    return qc

def inversion_test_circuit(x1, x2, reuploads):
    """
    Build the Inversion Test circuit for data points x1 and x2.

    Circuit: U(x1) · U(x2)†, measured in the computational basis.
    K(x1, x2) = P(measure |0...0⟩) = |⟨0|U(x2)†U(x1)|0⟩|²
    """
    n_pc = len(x1)
    U_x1 = quantum_feature(x1, reuploads)
    U_x2 = quantum_feature(x2, reuploads)
    kernel_c = U_x1.compose(U_x2.inverse(), range(n_pc))
    kernel_c.measure(range(n_pc), range(n_pc))
    return kernel_c

# ─────────────────────────────────────────────────────────────────────────────
# qIT kernel
# ─────────────────────────────────────────────────────────────────────────────

def get_kernel_element_qIT(x1, x2, n_shots=1000):
    """
    Estimate K_qIT(x1, x2) via quantum circuit simulation.

    Runs the Inversion Test circuit with n_shots measurements.
    Returns the empirical probability of the all-zeros outcome.
    Shot noise ≈ 1/√n_shots per element; the resulting kernel matrix
    may have small negative eigenvalues (handled by regularize_kernel).
    Uses GPU simulation if available, otherwise falls back to CPU.
    """
    kernel_c  = inversion_test_circuit(x1, x2, 3)
    simulator = QasmSimulator(device="GPU") if _USE_GPU else QasmSimulator()
    t_circuit = transpile(kernel_c, simulator)
    job       = simulator.run(t_circuit, shots=n_shots)
    counts    = job.result().get_counts()
    prob0     = counts.get("0" * len(x1), 0) / n_shots
    return prob0

def get_kernel_matrix_qIT(X1, X2, seed=None, kmethod="qIT",
                           n_shots=1000, qVS_subsamples=None, qVS_maxsize=None):
    """
    Compute or resume the qIT kernel matrix K[i,j] = K_qIT(X1[i], X2[j]).

    For the training matrix (X1 == X2), only the upper triangle is computed
    and mirrored (kernel is symmetric by definition of qIT).
    Intermediate results are saved after each diagonal entry to allow
    resuming interrupted computations.

    Cache key includes seed, which should be the composite seed
    (seed × 1000 + file_id) to ensure per-file cache isolation.
    """
    X1_size = len(X1)
    X2_size = len(X2)
    n_pc    = len(X1[0])
    split   = "train" if np.array_equal(X1, X2) else "test"

    start_t      = time.time()
    gram_matrix  = retrieve_interim_kernel_copy(kmethod, (X1_size, X2_size), seed, n_pc,
                                                split=split, qIT_shots=n_shots,
                                                qVS_subsamples=qVS_subsamples,
                                                qVS_maxsize=qVS_maxsize)

    if gram_matrix is not None and gram_matrix[-1, -1] > 0:
        return gram_matrix
    elif gram_matrix is None:
        gram_matrix = np.zeros((X1_size, X2_size))
        num_eval    = X1_size * X2_size
        indices     = product(range(X1_size), range(X2_size))
    else:
        next_i, next_j = find_kernel_entry_index(gram_matrix)
        assert next_i != -1, "Expected incomplete matrix but find_kernel_entry_index returned -1."
        num_eval = (X1_size - next_i) * X2_size - next_j
        indices  = chain(product([next_i], range(next_j, X2_size)),
                         product(range(next_i + 1, X1_size), range(0, X2_size)))

    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, False, kmethod,
                                         (X1_size, X2_size), split, seed, n_pc,
                                         qIT_shots=n_shots,
                                         qVS_subsamples=qVS_subsamples,
                                         qVS_maxsize=qVS_maxsize)

    progress = tqdm(indices, total=num_eval)
    for i, j in progress:
        start_t = time.time()
        progress.set_description(f"gram_matrix [{i}][{j}]")
        if split == "test" or (split == "train" and j >= i):
            gram_matrix[i][j] = get_kernel_element_qIT(X1[i], X2[j], n_shots)
            if j == i or n_pc > 8:
                save_interim_kernel_copy(gram_matrix, kmethod, (X1_size, X2_size),
                                          seed, n_pc, split=split, qIT_shots=n_shots,
                                          qVS_subsamples=qVS_subsamples,
                                          qVS_maxsize=qVS_maxsize)
                end_t = time.time()
                save_interim_kernel_calculation_time(end_t - start_t, False, kmethod,
                                                      (X1_size, X2_size), split, seed, n_pc,
                                                      qIT_shots=n_shots,
                                                      qVS_subsamples=qVS_subsamples,
                                                      qVS_maxsize=qVS_maxsize)

    # Mirror upper triangle to lower for the symmetric training matrix
    if split == "train":
        gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))

    save_interim_kernel_copy(gram_matrix, kmethod, (X1_size, X2_size),
                              seed, n_pc, split=split, qIT_shots=n_shots,
                              qVS_subsamples=qVS_subsamples, qVS_maxsize=qVS_maxsize)
    save_interim_kernel_calculation_time(0, True, kmethod, (X1_size, X2_size),
                                          split, seed, n_pc, qIT_shots=n_shots,
                                          qVS_subsamples=qVS_subsamples,
                                          qVS_maxsize=qVS_maxsize)
    return gram_matrix

# ─────────────────────────────────────────────────────────────────────────────
# qRM kernel
# ─────────────────────────────────────────────────────────────────────────────

def get_qRM_settings_list(seed, n_pc, n_settings):
    """
    Return a list of n_settings random measurement settings.
    Each setting is a list of n_pc local Haar-random 2×2 unitaries.
    Settings are cached on disk and reused across runs with the same seed.
    """
    qRM_settings_list = get_saved_qRM_settings(seed, n_pc, n_settings)
    if qRM_settings_list is None:
        qRM_settings_list = []
        for _ in range(n_settings):
            measurement_setting = [random_unitary(2, seed=seed) for _ in range(n_pc)]
            qRM_settings_list.append(measurement_setting)
        save_qRM_settings(qRM_settings_list, seed, n_pc, n_settings)
    return qRM_settings_list

def single_random_measurement_circuit(x, n_reuploads, measurement_setting):
    """
    Build the circuit: U(x) followed by local Haar-random unitaries.
    Measurement in the computational basis after rotation.
    """
    n_pc    = len(x)
    U_x     = quantum_feature(x, n_reuploads)
    kernel_c = U_x.copy()
    for i in range(n_pc):
        kernel_c.append(measurement_setting[i], [i])
    kernel_c.measure(range(n_pc), range(n_pc))
    return kernel_c

def get_single_random_measurement_results(x, measurement_setting, shots):
    """
    Run a single randomized measurement circuit for data point x.
    Returns a list of (bitstring_as_list, probability) tuples.
    """
    kernel_c  = single_random_measurement_circuit(x, 3, measurement_setting)
    # qRM currently uses CPU only; GPU support can be added analogously to qIT
    simulator = QasmSimulator()
    t_circuit = transpile(kernel_c, simulator)
    job       = simulator.run(t_circuit, shots=shots)
    counts    = job.result().get_counts()
    return [(list(k), v / shots) for k, v in counts.items()]

def get_random_measurements_results(x, qRM_settings_list, n_shots=8000):
    """Collect randomized measurement outcomes for x across all settings."""
    return [get_single_random_measurement_results(x, ms, n_shots)
            for ms in qRM_settings_list]

def get_dataset_randomized_measurements(X, dataset_index, seed, qRM_settings,
                                         qRM_shots, split):
    """
    Compute and cache randomized measurements for all points in X.
    Resumes from the last saved checkpoint if a partial result exists.
    """
    X_measurements = retrieve_interim_qRM_measurements(
        dataset_index, len(X), seed, len(X[0]), qRM_shots, len(qRM_settings), split
    )
    X_measurements = list(X_measurements) if X_measurements is not None else None

    if X_measurements is None:
        X_measurements    = []
        rml_progress_bar  = trange(0, len(X), position=0, leave=True)
    elif len(X_measurements) == len(X):
        return X_measurements
    else:
        rml_progress_bar = trange(len(X_measurements), len(X), position=0, leave=True)

    for i in rml_progress_bar:
        start_t = time.time()
        rml_progress_bar.set_description(f"Measuring point {i}")
        x_results = get_random_measurements_results(X[i], qRM_settings, qRM_shots)
        X_measurements.append(x_results)
        save_interim_qRM_measurements(dataset_index, X_measurements, len(X),
                                       seed, len(X[0]), qRM_shots, len(qRM_settings), split)
        end_t = time.time()
        save_interim_kernel_calculation_time(end_t - start_t, False, "qRM",
                                              len(X), split, seed, len(X[0]),
                                              qRM_shots=qRM_shots,
                                              qRM_settings=len(qRM_settings))
    return X_measurements

def get_kernel_matrix_qRM(X1, X2, seed=None, n_settings=8, n_shots=8000):
    """
    Compute the qRM kernel matrix via randomized measurements.

    Implements the cross-purity estimator of Eq. 7, arXiv:2108.01039.
    Error mitigation is applied: off-diagonal entries are normalized by
    the geometric mean of the corresponding diagonal purities.
    """
    X1_size, X2_size, n_pc = len(X1), len(X2), len(X1[0])
    split = "train" if np.array_equal(X1, X2) else "test"

    qRM_settings_list = get_qRM_settings_list(seed, n_pc, n_settings)
    gram_matrix = retrieve_interim_kernel_copy("qRM", (X1_size, X2_size), seed, n_pc,
                                               split=split, qRM_shots=n_shots,
                                               qRM_settings=n_settings)

    if gram_matrix is not None and is_kernel_complete("qRM", (X1_size, X2_size),
                                                       split, seed, n_pc, n_shots, n_settings):
        return gram_matrix

    # Apply mitigation if kernel is computed but mitigation has not been applied
    if gram_matrix is not None and (gram_matrix[-1, -1] != 0 and gram_matrix[-1, 1] != 1):
        start_t        = time.time()
        X1_measurements = get_dataset_randomized_measurements(X1, 1, seed, qRM_settings_list, n_shots, split)
        X2_measurements = (get_dataset_randomized_measurements(X2, 2, seed, qRM_settings_list, n_shots, split)
                           if split == "test" else X1_measurements)
        X1_purities = ([combine_randomized_measurements(x, x) for x in X1_measurements]
                       if split == "test" else None)
        X2_purities = ([combine_randomized_measurements(x, x) for x in X2_measurements]
                       if split == "test" else None)
        gram_matrix = apply_mitigation(gram_matrix, split,
                                        X1_purities=X1_purities, X2_purities=X2_purities)
        if split == "train":
            gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))
        save_interim_kernel_copy(gram_matrix, "qRM", (X1_size, X2_size), seed, n_pc,
                                  split=split, qRM_shots=n_shots, qRM_settings=n_settings)
        end_t = time.time()
        save_interim_kernel_calculation_time(end_t - start_t, True, "qRM",
                                              (X1_size, X2_size), split, seed, n_pc,
                                              qRM_shots=n_shots, qRM_settings=n_settings)
        return gram_matrix

    start_t = time.time()
    qRM_settings_list = get_qRM_settings_list(seed, n_pc, n_settings)
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, False, "qRM",
                                          (X1_size, X2_size), split, seed, n_pc,
                                          qRM_shots=n_shots, qRM_settings=n_settings)

    start_t         = time.time()
    X1_measurements = get_dataset_randomized_measurements(X1, 1, seed, qRM_settings_list, n_shots, split)
    X2_measurements = (get_dataset_randomized_measurements(X2, 2, seed, qRM_settings_list, n_shots, split)
                       if split == "test" else X1_measurements)

    if gram_matrix is None:
        gram_matrix = np.zeros((X1_size, X2_size))
        num_eval    = X1_size * X2_size
        indices     = product(range(X1_size), range(X2_size))
    else:
        next_i, next_j = find_kernel_entry_index(gram_matrix)
        num_eval = (X1_size - next_i) * X2_size - next_j
        indices  = chain(product([next_i], range(next_j, X2_size)),
                         product(range(next_i + 1, X1_size), range(0, X2_size)))

    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, False, "qRM",
                                          (X1_size, X2_size), split, seed, n_pc,
                                          qRM_shots=n_shots, qRM_settings=n_settings)

    km_progress = tqdm(indices, total=num_eval, position=0, leave=True)
    for i, j in km_progress:
        km_progress.set_description(f"gram_matrix RM [{i}][{j}]")
        if split == "test" or (split == "train" and j >= i):
            start_t = time.time()
            gram_matrix[i][j] = combine_randomized_measurements(
                X1_measurements[i], X2_measurements[j])
            if j == i or n_pc > 8:
                save_interim_kernel_copy(gram_matrix, "qRM", (X1_size, X2_size),
                                          seed, n_pc, split=split,
                                          qRM_shots=n_shots, qRM_settings=n_settings)
                end_t = time.time()
                save_interim_kernel_calculation_time(end_t - start_t, False, "qRM",
                                                      (X1_size, X2_size), split, seed, n_pc,
                                                      qRM_shots=n_shots, qRM_settings=n_settings)

    start_t     = time.time()
    X1_purities = ([combine_randomized_measurements(x, x) for x in X1_measurements]
                   if split == "test" else None)
    X2_purities = ([combine_randomized_measurements(x, x) for x in X2_measurements]
                   if split == "test" else None)
    gram_matrix = apply_mitigation(gram_matrix, split,
                                    X1_purities=X1_purities, X2_purities=X2_purities)
    if split == "train":
        gram_matrix = gram_matrix + gram_matrix.T - np.diag(np.diag(gram_matrix))

    save_interim_kernel_copy(gram_matrix, "qRM", (X1_size, X2_size), seed, n_pc,
                              split=split, qRM_shots=n_shots, qRM_settings=n_settings)
    end_t = time.time()
    save_interim_kernel_calculation_time(end_t - start_t, True, "qRM",
                                          (X1_size, X2_size), split, seed, n_pc,
                                          qRM_shots=n_shots, qRM_settings=n_settings)
    return gram_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Randomized measurement helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_exponential_hamming_matrix(A, B):
    """Compute (-2)^{-H(a,b)} for all (a,b) pairs; used in the cross-purity estimator."""
    powers = -np.count_nonzero(A[:, None, :] != B[None, :, :], axis=-1)
    return np.float_power(np.array(-2), powers)

def get_single_unitary_trace(probA, probB, strA, strB):
    """
    Estimate Tr[ρ_A ρ_B] for a single measurement setting via the
    cross-purity formula of arXiv:2108.01039.
    """
    prob_product              = np.outer(probA, probB)
    exponential_hamming       = get_exponential_hamming_matrix(strA, strB)
    return np.sum(np.multiply(exponential_hamming, prob_product))

def combine_randomized_measurements(x1_measurements, x2_measurements):
    """
    Combine randomized measurement results across all settings to estimate
    K_qRM(x1, x2) following Eq. 7 of arXiv:2108.01039.

    Returns the kernel element K(x1, x2) = 2^n · mean_M Tr[ρ_x1^M ρ_x2^M].
    """
    n_pc             = len(x1_measurements[0][0][0])
    traces_by_setting = []
    for x1_result, x2_result in zip(x1_measurements, x2_measurements):
        strA  = np.array(np.vstack(np.array(x1_result, dtype=object)[:, 0]))
        probA = np.array(x1_result, dtype=object)[:, 1]
        strB  = np.array(np.vstack(np.array(x2_result, dtype=object)[:, 0]))
        probB = np.array(x2_result, dtype=object)[:, 1]
        traces_by_setting.append(get_single_unitary_trace(probA, probB, strA, strB))
    return (2 ** n_pc) * np.mean(traces_by_setting)

def apply_mitigation(gram_matrix, split, X1_purities=None, X2_purities=None):
    """
    Apply error mitigation to the qRM kernel matrix.

    Train split: K_mitigated[i,j] = K[i,j] / sqrt(K[i,i] · K[j,j])
    Test split : K_mitigated[i,j] = K[i,j] / sqrt(purity_i · purity_j)
    """
    num_eval   = gram_matrix.shape[0] * gram_matrix.shape[1]
    indices    = product(range(gram_matrix.shape[0]), range(gram_matrix.shape[1]))
    em_progress = tqdm(indices, total=num_eval, position=0, leave=True)

    if split == "train":
        for i, j in em_progress:
            em_progress.set_description(f"Error mitigation [{i}][{j}]")
            if i != j:
                gram_matrix[i][j] /= np.sqrt(gram_matrix[i][i] * gram_matrix[j][j])
        for i in range(gram_matrix.shape[0]):
            gram_matrix[i][i] = 1.0
    else:
        for i, j in em_progress:
            em_progress.set_description(f"Error mitigation [{i}][{j}]")
            gram_matrix[i][j] /= np.sqrt(X1_purities[i] * X2_purities[j])
    return gram_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Cache I/O
# ─────────────────────────────────────────────────────────────────────────────

def _kernel_cache_path(kmethod, size, seed, n_pc, split,
                        qIT_shots=None, qRM_shots=None, qRM_settings=None,
                        qVS_subsamples=None, qVS_maxsize=None):
    """Construct the canonical cache file path for a given kernel configuration."""
    folder = os.path.join(ROOT_DIR, "InterimResults", kmethod, split)
    os.makedirs(folder, exist_ok=True)
    fname = f"dsize_{size}_seed_{seed}_n_pc_{n_pc}"
    if kmethod == "qIT":
        fname += f"_n_shots_{qIT_shots}"
    elif kmethod == "qRM":
        fname += f"_n_shots_{qRM_shots}_n_settings_{qRM_settings}"
    elif kmethod == "qVS":
        fname += f"_n_subsamples_{qVS_subsamples}_n_maxsize_{qVS_maxsize}"
    return os.path.join(folder, fname + ".npy")

def save_interim_kernel_copy(interimKernelCopy, kmethod, size, seed, n_pc,
                              split=None, qIT_shots=None, qRM_shots=None,
                              qRM_settings=None, qVS_subsamples=None, qVS_maxsize=None):
    path = _kernel_cache_path(kmethod, size, seed, n_pc, split, qIT_shots,
                               qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize)
    np.save(path, interimKernelCopy)

def retrieve_interim_kernel_copy(kmethod, size, seed, n_pc, split=None,
                                  qIT_shots=None, qRM_shots=None, qRM_settings=None,
                                  qVS_subsamples=None, qVS_maxsize=None):
    """Return the cached kernel matrix, or None if no cache exists."""
    path = _kernel_cache_path(kmethod, size, seed, n_pc, split, qIT_shots,
                               qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize)
    return np.load(path) if os.path.exists(path) else None

def find_kernel_entry_index(interimKernelCopy):
    """
    Find the (i, j) index of the next kernel entry to compute.
    Returns (-1, -1) if the matrix is fully computed.
    """
    shape         = interimKernelCopy.shape
    nonzero_entries = np.argwhere(interimKernelCopy != 0)
    if nonzero_entries.size == 0:
        return 0, 0
    last = nonzero_entries[-1]
    if last[1] == shape[1] - 1:
        return (-1, -1) if last[0] == shape[0] - 1 else (last[0] + 1, 0)
    return (last[0], last[1] + 1)

def save_interim_qRM_measurements(dataset_index, qRMmeasurements, size, seed,
                                   n_pc, qRM_shots, qRM_settings, split):
    """Persist randomized measurement outcomes for dataset X_{dataset_index}."""
    folder = os.path.join(ROOT_DIR, "InterimResults", "qRM", "Measurements")
    os.makedirs(folder, exist_ok=True)
    fname = (f"{split}_X{dataset_index}_dsize_{size}_seed_{seed}_n_pc_{n_pc}"
             f"_n_shots_{qRM_shots}_n_settings_{qRM_settings}.npy")
    np.save(os.path.join(folder, fname),
            np.array(qRMmeasurements, dtype=object), allow_pickle=True)

def retrieve_interim_qRM_measurements(dataset_index, size, seed, n_pc,
                                       qRM_shots, qRM_settings, split):
    """Load cached randomized measurements, or return None if absent."""
    folder = os.path.join(ROOT_DIR, "InterimResults", "qRM", "Measurements")
    fname  = (f"{split}_X{dataset_index}_dsize_{size}_seed_{seed}_n_pc_{n_pc}"
              f"_n_shots_{qRM_shots}_n_settings_{qRM_settings}.npy")
    path   = os.path.join(folder, fname)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None

def save_qRM_settings(qRM_settings_list, seed, n_pc, qRM_settings):
    """Persist the random measurement settings for reproducibility."""
    folder = os.path.join(ROOT_DIR, "InterimResults", "qRM", "Measurements")
    os.makedirs(folder, exist_ok=True)
    fname = f"settings_seed_{seed}_n_pc_{n_pc}_n_settings_{qRM_settings}.npy"
    np.save(os.path.join(folder, fname),
            np.array(qRM_settings_list, dtype=object), allow_pickle=True)

def get_saved_qRM_settings(seed, n_pc, qRM_settings):
    """Load previously saved random measurement settings, or return None."""
    folder = os.path.join(ROOT_DIR, "InterimResults", "qRM", "Measurements")
    fname  = f"settings_seed_{seed}_n_pc_{n_pc}_n_settings_{qRM_settings}.npy"
    path   = os.path.join(folder, fname)
    return np.load(path, allow_pickle=True) if os.path.exists(path) else None

# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────

def _timing_path(kmethod, size, split, seed, n_pc,
                  qIT_shots=None, qRM_shots=None, qRM_settings=None,
                  qVS_subsamples=None, qVS_maxsize=None):
    folder = os.path.join(".", "InterimResults", kmethod, split)
    os.makedirs(folder, exist_ok=True)
    fname = f"kernel_calculation_times_dsize_{size}_seed_{seed}_n_pc_{n_pc}"
    if kmethod == "qIT":
        fname += f"_n_shots_{qIT_shots}"
    elif kmethod == "qRM":
        fname += f"_n_shots_{qRM_shots}_n_settings_{qRM_settings}"
    elif kmethod == "qVS":
        fname += f"_n_subsamples_{qVS_subsamples}_maxsize_{qVS_maxsize}"
    return os.path.join(folder, fname + ".npy")

def save_interim_kernel_calculation_time(calc_time, complete, kmethod, size,
                                          split, seed, n_pc, qIT_shots=None,
                                          qRM_shots=None, qRM_settings=None,
                                          qVS_subsamples=None, qVS_maxsize=None):
    """Accumulate elapsed computation time in the timing cache file."""
    path = _timing_path(kmethod, size, split, seed, n_pc, qIT_shots,
                         qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize)
    if not os.path.exists(path):
        np.save(path, np.array([calc_time, complete]))
    else:
        t = np.load(path)
        if t.size == 0:
            np.save(path, np.array([calc_time, complete]))
        elif not t[1]:
            t[0] += calc_time
            t[1]  = complete
            np.save(path, t)

def retrieve_interim_kernel_calculation_time(kmethod, size, split, seed, n_pc,
                                              qIT_shots=None, qRM_shots=None,
                                              qRM_settings=None,
                                              qVS_subsamples=None, qVS_maxsize=None):
    """Return accumulated computation time from the timing cache, or 0."""
    path = _timing_path(kmethod, size, split, seed, n_pc, qIT_shots,
                         qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize)
    if not os.path.exists(path):
        return 0
    t = np.load(path)
    return t[0] if t.size > 0 else 0

def is_kernel_complete(kmethod, size, split, seed, n_pc, qIT_shots=None,
                        qRM_shots=None, qRM_settings=None,
                        qVS_subsamples=None, qVS_maxsize=None):
    """Return True if the timing cache indicates the kernel computation is done."""
    path = _timing_path(kmethod, size, split, seed, n_pc, qIT_shots,
                         qRM_shots, qRM_settings, qVS_subsamples, qVS_maxsize)
    if not os.path.exists(path):
        return False
    t = np.load(path)
    return bool(t[1]) if t.size > 0 else False