"""Shared deterministic inputs and compact output records for Stage 4."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


GENERATOR_NAME = "splitmix64-uniform01-v1"
OUTPUT_HASH_NAME = "fnv1a64-fp64-le-v1"
QUERY_STREAM_TAG = 0xD1B54A32D192ED03
UINT64_MASK = (1 << 64) - 1
INVERSE_TWO_TO_53 = 1.0 / 9007199254740992.0
GENERATION_CHUNK_VALUES = 4 * 1024 * 1024


def positive_int32(text: str) -> int:
    value = int(text)
    if value <= 0 or value > 2_147_483_647:
        raise ValueError("value must be a positive 32-bit integer")
    return value


def uint64_seed(text: str) -> int:
    value = int(text)
    if value < 0 or value > UINT64_MASK:
        raise ValueError("seed must be an unsigned 64-bit integer")
    return value


def _generate_matrix(rows: int, dimension: int, stream_seed: int) -> np.ndarray:
    count = rows * dimension
    if count > sys.maxsize // np.dtype(np.float64).itemsize:
        raise ValueError("synthetic matrix is too large for this host")

    matrix = np.empty(count, dtype=np.float64)
    for start in range(0, count, GENERATION_CHUNK_VALUES):
        end = min(start + GENERATION_CHUNK_VALUES, count)
        values = np.arange(start, end, dtype=np.uint64)
        values += np.uint64(stream_seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values ^= values >> 30
        values *= np.uint64(0xBF58476D1CE4E5B9)
        values ^= values >> 27
        values *= np.uint64(0x94D049BB133111EB)
        values ^= values >> 31
        matrix[start:end] = (values >> 11).astype(np.float64)
        matrix[start:end] *= INVERSE_TWO_TO_53
    return matrix.reshape(rows, dimension)


def generate_inputs(
    data_rows: int,
    query_rows: int,
    dimension: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = _generate_matrix(data_rows, dimension, seed)
    queries = _generate_matrix(
        query_rows,
        dimension,
        seed ^ QUERY_STREAM_TAG,
    )
    return data, queries


def output_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.dtype("<f8")).reshape(-1)
    raw = contiguous.view(np.uint8)
    hash_value = 14695981039346656037
    for byte in raw:
        hash_value ^= int(byte)
        hash_value = (hash_value * 1099511628211) & UINT64_MASK
    return f"0x{hash_value:016x}"


def write_output(path: str | Path, values: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            handle.write(f"{float(value):.17g}\n")
