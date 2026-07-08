#!/usr/bin/env python3
"""Dense `.data` / `.data.zst` readers for Stage 2 baseline adapters."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import zstandard as zstd


def _is_zstd(path: Path) -> bool:
    return path.name.endswith(".zst")


def _read_zstd_header_and_payload(path: Path) -> tuple[list[str], bytes]:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as file_obj:
        with dctx.stream_reader(file_obj) as reader:
            buffered = io.BufferedReader(reader)
            header = buffered.readline().decode("utf-8").strip().split()
            payload = buffered.read()
    return header, payload


def _read_zstd_bytes(path: Path) -> bytes:
    dctx = zstd.ZstdDecompressor()
    with path.open("rb") as file_obj:
        with dctx.stream_reader(file_obj) as reader:
            return reader.read()


def read_dense_header(path: str | Path) -> tuple[int, int]:
    path = Path(path)
    if _is_zstd(path):
        header, _ = _read_zstd_header_and_payload(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip().split()
    if len(header) != 2:
        raise ValueError(f"{path}: expected first line '<rows> <dim>'")
    return int(header[0]), int(header[1])


def read_dense_matrix(path: str | Path, dtype: np.dtype) -> np.ndarray:
    path = Path(path)
    if _is_zstd(path):
        header, payload = _read_zstd_header_and_payload(path)
        if len(header) != 2:
            raise ValueError(f"{path}: expected first line '<rows> <dim>'")
        rows, dim = int(header[0]), int(header[1])
        values = np.fromstring(payload, sep=" ", dtype=dtype, count=rows * dim)
    else:
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


def read_dense_vector(path: str | Path, dtype: np.dtype) -> np.ndarray:
    path = Path(path)
    if _is_zstd(path):
        payload = _read_zstd_bytes(path)
        return np.fromstring(payload, sep=" ", dtype=dtype).reshape(-1)
    return np.loadtxt(path, dtype=dtype).reshape(-1)
