#!/usr/bin/env python3
"""Stage 1-p1 cuML adapter for exact raw Gaussian SVM workloads.

The adapter follows the timing contract in GPU-kernel-density-exact
docs/stage1-p1/goal.md.  Disk I/O, query loading, and Scott/scalar coordinate
scaling happen before the timer.  The measured `end_to_end_in_memory` scope
starts from scaled CPU arrays and includes host-to-device transfer, cuML
KernelDensity construction and fit, score_samples, GPU denormalization back to
the raw kernel-sum convention, and host output materialization.  Writing the
.out file happens after the timer.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np


CUML_BANDWIDTH = 1.0 / math.sqrt(2.0)
TIMING_SCOPE = "end_to_end_in_memory"
METHOD = "cuML-KDE"
REQUIRED_CUML_VERSION_PREFIX = "26.06"


def read_matrix(path: str | Path, dtype: np.dtype) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"{path}: expected first line '<rows> <dim>'")
        rows, dim = int(header[0]), int(header[1])
        values = np.fromfile(handle, sep=" ", dtype=dtype, count=rows * dim)

    if values.size != rows * dim:
        raise ValueError(
            f"{path}: expected {rows * dim} numeric values, found {values.size}"
        )
    return values.reshape(rows, dim)


def read_weights(path: str | Path, expected_rows: int, dtype: np.dtype) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype).reshape(-1)
    if values.size == expected_rows + 1 and int(values[0]) == expected_rows:
        values = values[1:]
    if values.size != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} weights, found {values.size}")
    return values


def read_reference(path: str | Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"{path}: reference output is empty")
    return values


def scalar_gamma_coefficients(data: np.ndarray, gamma: float) -> np.ndarray:
    if not np.isfinite(gamma) or gamma < 0:
        raise ValueError("--gamma must be finite and non-negative")
    return np.full(data.shape[1], np.sqrt(gamma), dtype=data.dtype)


def scott_diag_coefficients(data: np.ndarray, b: float) -> np.ndarray:
    if not np.isfinite(b) or b <= 0:
        raise ValueError("--scott-diag/--b must be positive")

    rows, dim = data.shape
    n_factor = float(rows) ** (-1.0 / float(dim + 4))
    std = data.std(axis=0, ddof=0)
    if np.any(std <= 0):
        bad = np.nonzero(std <= 0)[0]
        raise ValueError(
            "Scott diagonal scale is undefined for zero-variance dimensions: "
            + ", ".join(str(int(i)) for i in bad)
        )

    h = b * std * n_factor
    gamma = 1.0 / (2.0 * h * h)
    return np.sqrt(gamma).astype(data.dtype, copy=False)


def scale_coefficients(data: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.gamma is not None:
        return scalar_gamma_coefficients(data, args.gamma)
    return scott_diag_coefficients(data, args.scott_diag)


def scale_description(args: argparse.Namespace) -> str:
    if args.gamma is not None:
        return f"gamma={args.gamma}"
    return f"scott_diag_b={args.scott_diag}"


def gaussian_norm(dim: int, bandwidth: float) -> float:
    return 1.0 / (((2.0 * math.pi) ** (0.5 * dim)) * (bandwidth**dim))


def import_cuml():
    try:
        import cupy as cp
        import cuml
        from cuml.neighbors import KernelDensity
    except Exception as exc:
        raise RuntimeError(
            "Could not import cuML/CuPy. Activate the Stage 1-p1 cuML "
            "environment before running this adapter."
        ) from exc
    version = getattr(cuml, "__version__", "")
    if not str(version).startswith(REQUIRED_CUML_VERSION_PREFIX):
        raise RuntimeError(
            f"This Stage 1-p1 adapter requires cuML {REQUIRED_CUML_VERSION_PREFIX}.x, "
            f"but imported cuML version {version!r}."
        )
    return cp, cuml, KernelDensity


def fit_kde(KernelDensity, data_gpu, weights_gpu):
    kde = KernelDensity(
        kernel="gaussian",
        metric="euclidean",
        bandwidth=CUML_BANDWIDTH,
    )
    if weights_gpu is None:
        return kde.fit(data_gpu, convert_dtype=False)
    return kde.fit(data_gpu, sample_weight=weights_gpu, convert_dtype=False)


def to_cupy(cp, value):
    if hasattr(value, "to_output"):
        return value.to_output("cupy")
    return cp.asarray(value)


def evaluate_raw_sum(cp, kde, query_gpu, log_multiplier: float):
    log_density = kde.score_samples(query_gpu, convert_dtype=False)
    log_density_gpu = to_cupy(cp, log_density)
    return cp.exp(log_density_gpu + log_multiplier)


def run_pipeline(
    cp,
    KernelDensity,
    data_scaled: np.ndarray,
    query_scaled: np.ndarray,
    weights: Optional[np.ndarray],
    sum_weights: float,
) -> np.ndarray:
    data_gpu = cp.asarray(data_scaled)
    query_gpu = cp.asarray(query_scaled)
    weights_gpu = None if weights is None else cp.asarray(weights)

    kde = fit_kde(KernelDensity, data_gpu, weights_gpu)

    norm = gaussian_norm(data_scaled.shape[1], CUML_BANDWIDTH)
    log_multiplier = math.log(sum_weights / norm)
    raw_gpu = evaluate_raw_sum(cp, kde, query_gpu, log_multiplier)

    return cp.asnumpy(raw_gpu).reshape(-1)


def write_vector(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in np.asarray(values).reshape(-1):
            handle.write(f"{float(value):.17g}\n")


def compute_errors(output: np.ndarray, reference_path: Optional[str]) -> tuple[str, str]:
    if reference_path is None:
        return "", ""

    reference = read_reference(reference_path)
    output64 = np.asarray(output, dtype=np.float64).reshape(-1)
    if reference.shape != output64.shape:
        raise ValueError(
            f"{reference_path}: reference shape {reference.shape} does not match "
            f"output shape {output64.shape}"
        )

    abs_err = np.abs(output64 - reference)
    denom = np.abs(reference)
    rel_err = np.empty_like(abs_err)
    nonzero = denom > 0
    rel_err[nonzero] = abs_err[nonzero] / denom[nonzero]
    rel_err[~nonzero] = np.where(abs_err[~nonzero] == 0, 0.0, np.inf)
    return f"{float(abs_err.max()):.17g}", f"{float(rel_err.max()):.17g}"


def append_summary(path: str | Path, row: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "machine",
        "method",
        "cuml_package",
        "source_commit",
        "workload",
        "precision",
        "timing_mode",
        "timing_scope",
        "runtime_s",
        "qps",
        "max_abs_err",
        "max_rel_err",
        "status",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def precision_label(dtype: np.dtype) -> str:
    if dtype == np.dtype("float64"):
        return "FP64"
    if dtype == np.dtype("float32"):
        return "FP32"
    return str(dtype)


def run_svm(args: argparse.Namespace) -> int:
    dtype = np.dtype(args.dtype)
    cp, cuml, KernelDensity = import_cuml()

    load_start = time.perf_counter()
    data = read_matrix(args.data, dtype)
    query = read_matrix(args.query, dtype)
    if query.shape[1] != data.shape[1]:
        raise ValueError(
            f"dimension mismatch: data dim={data.shape[1]}, query dim={query.shape[1]}"
        )
    coeff = scale_coefficients(data, args)
    data_scaled = data * coeff
    query_scaled = query * coeff

    weights = None
    if args.weights is not None:
        weights = read_weights(args.weights, data.shape[0], dtype)
        sum_weights = float(weights.sum(dtype=np.float64))
    else:
        sum_weights = float(data.shape[0])
    if not np.isfinite(sum_weights) or sum_weights <= 0:
        raise ValueError("sum of weights must be finite and positive")
    load_elapsed = time.perf_counter() - load_start

    if args.timing_mode == "warm":
        _ = run_pipeline(cp, KernelDensity, data_scaled, query_scaled, weights, sum_weights)

    start = time.perf_counter()
    output = run_pipeline(cp, KernelDensity, data_scaled, query_scaled, weights, sum_weights)
    elapsed = time.perf_counter() - start

    output_path = Path(args.output)
    write_vector(output_path, output)

    query_count = int(query.shape[0])
    qps = float(query_count) / elapsed if elapsed > 0 else float("inf")
    max_abs_err, max_rel_err = compute_errors(output, args.reference)
    status = "ok"

    row = {
        "machine": args.machine,
        "method": METHOD,
        "cuml_package": getattr(cuml, "__version__", "unknown"),
        "source_commit": args.source_commit,
        "workload": args.workload,
        "precision": precision_label(dtype),
        "timing_mode": args.timing_mode,
        "timing_scope": TIMING_SCOPE,
        "runtime_s": f"{elapsed:.9f}",
        "qps": f"{qps:.9f}",
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "status": status,
    }
    if args.summary_csv is not None:
        append_summary(args.summary_csv, row)

    print("Mode: svm")
    print(f"Method: {METHOD}")
    print(f"cuML package: {row['cuml_package']}")
    print("KernelDensity kernel: gaussian")
    print("KernelDensity metric: euclidean")
    print(f"KernelDensity bandwidth: {CUML_BANDWIDTH:.17g}")
    print(f"Data: size={data.shape[0]}, dim={data.shape[1]}")
    print(f"Query count: {query_count}")
    print(f"Precision: {row['precision']}")
    print(f"Kernel scale: {scale_description(args)}")
    print(f"Load/preprocess time: {load_elapsed:.6f} seconds")
    print(f"Timing mode: {args.timing_mode}")
    print(f"Timing scope: {TIMING_SCOPE}")
    print(f"Elapsed time: {elapsed:.6f} seconds")
    print(f"Method {METHOD}: {qps:.6f} Queries/sec")
    print(f"timing_scope: {TIMING_SCOPE}")
    print(f"execution_seconds: {elapsed:.9f}")
    print(f"query_count: {query_count}")
    print(f"qps: {qps:.9f}")
    if args.reference is not None:
        print(f"max_abs_err: {max_abs_err}")
        print(f"max_rel_err: {max_rel_err}")
    print(f"Output: {output_path}")
    if args.summary_csv is not None:
        print(f"Summary CSV: {args.summary_csv}")
    return 0


def add_scale_args(parser: argparse.ArgumentParser) -> None:
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument("--gamma", type=float, help="Scalar Gaussian scale")
    scale_group.add_argument(
        "--scott-diag",
        "--b",
        dest="scott_diag",
        type=float,
        help="Diagonal Scott multiplier b",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1-p1 cuML exact raw Gaussian SVM adapter."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    svm = subparsers.add_parser("svm", help="Point-query SVM/KAQ workload")
    svm.add_argument("data", help="Exact KDE .data matrix file")
    svm.add_argument("query", help="Exact KDE .data query matrix file")
    svm.add_argument("output", help="Output .out vector file")
    add_scale_args(svm)
    svm.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    svm.add_argument("--weights", default=None, help="Optional one-value-per-row weights")
    svm.add_argument("--reference", default=None, help="Optional reference .out file")
    svm.add_argument("--summary-csv", default=None, help="Optional summary CSV to append")
    svm.add_argument(
        "--timing-mode",
        choices=("cold_start", "warm"),
        default="cold_start",
        help="cold_start runs one measured pipeline; warm runs one untimed full "
        "pipeline before the measured full pipeline.",
    )
    svm.add_argument("--machine", default="", help="Machine label for summary CSV")
    svm.add_argument("--workload", default="", help="Workload label for summary CSV")
    svm.add_argument(
        "--source-commit",
        default="",
        help="cuML source commit label for summary CSV",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "svm":
        return run_svm(args)
    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
