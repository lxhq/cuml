#!/usr/bin/env python3
"""cuML adapter for exact raw Gaussian kernel aggregation.

The script mirrors the kernel-scale convention in GPU-kernel-density-exact:
it embeds scalar gamma or diagonal Scott scaling into both data and query
coordinates, then evaluates

    F(q) = sum_x w_x * exp(-||q - x||^2)

with RAPIDS cuML KernelDensity. cuML returns normalized log-density values, so
the adapter de-normalizes the result back to the raw sum convention before
writing output and before correctness comparisons.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np


CUML_BANDWIDTH = 1.0 / math.sqrt(2.0)


def read_matrix(path: str | Path, dtype: np.dtype) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError(f"{path}: expected first line '<rows> <dim>'")
        rows, dim = int(header[0]), int(header[1])
        values = np.fromfile(f, sep=" ", dtype=dtype, count=rows * dim)

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


def batched_ranges(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def write_vector(path: str | Path, values: Iterable[np.ndarray]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for batch in values:
            for value in np.asarray(batch).reshape(-1):
                f.write(f"{float(value):.17g}\n")


def write_grid(path: str | Path, values: np.ndarray) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in values:
            f.write(" ".join(f"{float(v):.17g}" for v in row))
            f.write("\n")


def import_cuml():
    try:
        import cupy as cp
        from cuml.neighbors import KernelDensity
    except Exception as exc:
        raise RuntimeError(
            "Could not import cuML/CuPy. Use the local Stage 0 environment, e.g.:\n"
            "  /home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/"
            "venvs/cuml-kde/bin/python scripts/cuml_exact_kde.py ..."
        ) from exc
    return cp, KernelDensity


def fit_kde(KernelDensity, data_gpu, weights_gpu):
    kde = KernelDensity(
        kernel="gaussian",
        metric="euclidean",
        bandwidth=CUML_BANDWIDTH,
    )
    if weights_gpu is None:
        return kde.fit(data_gpu, convert_dtype=False)
    return kde.fit(data_gpu, sample_weight=weights_gpu, convert_dtype=False)


def timed_score_samples(cp, kde, query_gpu):
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    log_density = kde.score_samples(query_gpu, convert_dtype=False)
    end.record()
    end.synchronize()
    elapsed_s = cp.cuda.get_elapsed_time(start, end) / 1000.0
    return log_density, elapsed_s


def evaluate_batches(
    cp,
    kde,
    query_gpu,
    batch_size: int,
    multiplier: float,
    warmup: bool,
) -> Tuple[list[np.ndarray], float, int]:
    total = int(query_gpu.shape[0])
    if warmup and total > 0:
        warmup_end = min(batch_size, total)
        _, _ = timed_score_samples(cp, kde, query_gpu[:warmup_end])

    output_batches: list[np.ndarray] = []
    elapsed = 0.0
    for start, end in batched_ranges(total, batch_size):
        log_density, batch_elapsed = timed_score_samples(cp, kde, query_gpu[start:end])
        elapsed += batch_elapsed
        raw_gpu = cp.exp(log_density) * multiplier
        output_batches.append(cp.asnumpy(raw_gpu).reshape(-1))
    return output_batches, elapsed, total


def run_svm(args, cp, KernelDensity, data: np.ndarray, coeff: np.ndarray) -> Tuple[float, int]:
    query = read_matrix(args.query, data.dtype)
    if query.shape[1] != data.shape[1]:
        raise ValueError(
            f"dimension mismatch: data dim={data.shape[1]}, query dim={query.shape[1]}"
        )

    weights = None
    if args.weights is not None:
        weights = read_weights(args.weights, data.shape[0], data.dtype)
        sum_weights = float(weights.sum(dtype=np.float64))
    else:
        sum_weights = float(data.shape[0])

    data_scaled = data * coeff
    query_scaled = query * coeff

    data_gpu = cp.asarray(data_scaled)
    query_gpu = cp.asarray(query_scaled)
    weights_gpu = None if weights is None else cp.asarray(weights)
    cp.cuda.Stream.null.synchronize()

    kde = fit_kde(KernelDensity, data_gpu, weights_gpu)
    cp.cuda.Stream.null.synchronize()

    norm = gaussian_norm(data.shape[1], CUML_BANDWIDTH)
    output_batches, elapsed, query_count = evaluate_batches(
        cp,
        kde,
        query_gpu,
        args.batch_size,
        sum_weights / norm,
        args.warmup,
    )
    write_vector(args.output, output_batches)
    return elapsed, query_count


def make_kdv_queries(data: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if data.shape[1] != 2:
        raise ValueError(f"KDV mode requires 2D data, found dim={data.shape[1]}")

    row_l = float(data[:, 0].min())
    row_u = float(data[:, 0].max())
    col_l = float(data[:, 1].min())
    col_u = float(data[:, 1].max())
    row_incr = 0.0 if rows <= 1 else (row_u - row_l) / float(rows - 1)
    col_incr = 0.0 if cols <= 1 else (col_u - col_l) / float(cols - 1)

    row_coords = row_l + np.arange(rows, dtype=data.dtype) * row_incr
    col_coords = col_l + np.arange(cols, dtype=data.dtype) * col_incr
    query = np.empty((rows * cols, 2), dtype=data.dtype)
    query[:, 0] = np.repeat(row_coords, cols)
    query[:, 1] = np.tile(col_coords, rows)
    return query


def run_kdv(args, cp, KernelDensity, data: np.ndarray, coeff: np.ndarray) -> Tuple[float, int]:
    weights = None
    if args.weights is not None:
        weights = read_weights(args.weights, data.shape[0], data.dtype)
        sum_weights = float(weights.sum(dtype=np.float64))
    else:
        sum_weights = float(data.shape[0])

    data_scaled = data * coeff
    query_scaled = make_kdv_queries(data, args.rows, args.cols) * coeff

    data_gpu = cp.asarray(data_scaled)
    query_gpu = cp.asarray(query_scaled)
    weights_gpu = None if weights is None else cp.asarray(weights)
    cp.cuda.Stream.null.synchronize()

    kde = fit_kde(KernelDensity, data_gpu, weights_gpu)
    cp.cuda.Stream.null.synchronize()

    norm = gaussian_norm(data.shape[1], CUML_BANDWIDTH)
    output_batches, elapsed, query_count = evaluate_batches(
        cp,
        kde,
        query_gpu,
        args.batch_size,
        sum_weights / norm,
        args.warmup,
    )

    output = np.concatenate(output_batches).reshape(args.rows, args.cols)
    write_grid(args.output, output)
    return elapsed, query_count


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("data", help="Exact KDE .data matrix file")
    parser.add_argument("output", help="Output file")
    scale_group = parser.add_mutually_exclusive_group(required=True)
    scale_group.add_argument("--gamma", type=float, help="Scalar Gaussian scale")
    scale_group.add_argument(
        "--scott-diag",
        "--b",
        dest="scott_diag",
        type=float,
        help="Diagonal Scott multiplier b",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--weights", default=None, help="Optional one-value-per-row weights")
    parser.add_argument(
        "--no-warmup",
        action="store_false",
        dest="warmup",
        help="Include first-call overhead in elapsed time.",
    )
    parser.set_defaults(warmup=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact cuML-KDE baseline for raw Gaussian SVM/KDV tasks."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    svm = subparsers.add_parser("svm", help="Point-query KAQ/SVM-style workload")
    svm.add_argument("query", help="Exact KDE .data query matrix file")
    add_common_args(svm)

    kdv = subparsers.add_parser("kdv", help="2D KDV grid workload")
    kdv.add_argument("--rows", type=int, required=True)
    kdv.add_argument("--cols", type=int, required=True)
    add_common_args(kdv)

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.mode == "kdv" and (args.rows <= 0 or args.cols <= 0):
        parser.error("--rows and --cols must be positive")
    return args


def main() -> int:
    args = parse_args()
    dtype = np.dtype(args.dtype)

    cp, KernelDensity = import_cuml()

    load_start = time.perf_counter()
    data = read_matrix(args.data, dtype)
    coeff = scale_coefficients(data, args)
    load_elapsed = time.perf_counter() - load_start

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "svm":
        elapsed, query_count = run_svm(args, cp, KernelDensity, data, coeff)
    elif args.mode == "kdv":
        elapsed, query_count = run_kdv(args, cp, KernelDensity, data, coeff)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    qps = float(query_count) / elapsed if elapsed > 0 else float("inf")
    print(f"Mode: {args.mode}")
    print("Method: cuML-KDE")
    print("KernelDensity kernel: gaussian")
    print("KernelDensity metric: euclidean")
    print(f"KernelDensity bandwidth: {CUML_BANDWIDTH:.17g}")
    print(f"Data: size={data.shape[0]}, dim={data.shape[1]}")
    print(f"Kernel scale: {scale_description(args)}")
    print(f"Query count: {query_count}")
    print(f"Load/preprocess time: {load_elapsed:.6f} seconds")
    print(f"Elapsed time: {elapsed:.6f} seconds")
    print(f"Method cuML-KDE: {qps:.6f} Queries/sec")
    print("timing_scope: score_samples_only")
    print(f"execution_seconds: {elapsed:.9f}")
    print(f"query_count: {query_count}")
    print(f"qps: {qps:.9f}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
