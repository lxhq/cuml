#!/usr/bin/env python3
"""Stage 0.1 local output-preservation runner for cuML-KDE."""

from __future__ import annotations

import os
import sys
from pathlib import Path


EXACT_P1_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/GPU-kernel-density-exact-stage0-p1")
sys.path.insert(0, str(EXACT_P1_ROOT / "scripts"))

from stage0p1_output_utils import Attempt, run_attempts, run_env_with_cuda


CUML_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/cuml")
HELPER = CUML_ROOT / "scripts/cuml_exact_kde.py"
PYTHON_BIN = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/venvs/kde-baselines/bin/python")
CUDA_HOME = Path("/usr/local/cuda-12.6")
RUN_ROOT = Path(
    os.environ.get(
        "STAGE0P1_RUN_ROOT",
        "/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0-p1/local-3080ti",
    )
)
DATA_ROOT = Path("/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
MACHINE = "local-3080ti"
METHOD = "cuML-KDE"
METHOD_TOKEN = "cuml-kde"
SCALE = "scott_diag b=1"
SCALE_VALUE = "1"
TIMEOUT_SECONDS = 3600
SVM_BATCH_SIZE = 4096
KDV_BATCH_SIZE = 8192
PAIR_BUFFER_BYTES = 4 * 1024 * 1024 * 1024

WORKLOADS = (
    ("svm_susy", "svm", DATA_ROOT / "susy/SUSY_X.data", DATA_ROOT / "susy/SUSY_qSet.data", None, None, 10000),
    ("svm_home", "svm", DATA_ROOT / "home/HT_Sensor_dataset_X.data", DATA_ROOT / "home/HT_Sensor_dataset_qSet.data", None, None, 10000),
    ("svm_miniboone", "svm", DATA_ROOT / "miniboone/MiniBooNE_X.data", DATA_ROOT / "miniboone/MiniBooNE_qSet.data", None, None, 10000),
    ("kdv_home_visual", "kdv", DATA_ROOT / "home_visualization/HT_Sensor_dataset_vis_X.data", None, 1920, 2560, 4915200),
    ("kdv_susy_visual", "kdv", DATA_ROOT / "susy_visualization/SUSY_vis_X.data", None, 1920, 2560, 4915200),
)
PRECISIONS = (("FP64", "float64"), ("FP32", "float32"))


def venv_library_paths() -> list[Path]:
    site_packages = PYTHON_BIN.parents[1] / "lib/python3.10/site-packages"
    paths = sorted(site_packages.glob("lib*/lib64"))
    paths.extend(sorted((site_packages / "nvidia").glob("*/lib")))
    return paths


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
    return int(first[0]), int(first[1])


def batch_size_for(mode: str, data_path: Path) -> int:
    data_rows, _ = matrix_shape(data_path)
    cap = SVM_BATCH_SIZE if mode == "svm" else KDV_BATCH_SIZE
    memory_limited = max(1, PAIR_BUFFER_BYTES // (data_rows * 8))
    return int(min(cap, memory_limited))


def output_rel(precision: str, workload: str, rows: int | None, cols: int | None) -> Path:
    token = precision.lower()
    if rows is not None and cols is not None:
        name = f"{MACHINE}_cuml-kde_{token}_{workload}_{rows}x{cols}_scott_diag_b1.out"
    else:
        name = f"{MACHINE}_cuml-kde_{token}_{workload}_scott_diag_b1.out"
    return Path("outputs") / METHOD_TOKEN / token / name


def command_for(mode: str, data_path: Path, query_path: Path | None, rows: int | None, cols: int | None, dtype: str, out: Path) -> tuple[str, ...]:
    command: list[str] = [str(PYTHON_BIN), str(HELPER), mode]
    if mode == "svm":
        assert query_path is not None
        command.extend([str(query_path), str(data_path), str(out)])
    else:
        assert rows is not None and cols is not None
        command.extend(["--rows", str(rows), "--cols", str(cols), str(data_path), str(out)])
    command.extend([
        "--scott-diag",
        SCALE_VALUE,
        "--batch-size",
        str(batch_size_for(mode, data_path)),
        "--dtype",
        dtype,
    ])
    return tuple(command)


def planned_attempts() -> list[Attempt]:
    env = run_env_with_cuda(CUDA_HOME)
    env["PYTHONNOUSERSITE"] = "1"
    env["PATH"] = f"{PYTHON_BIN.parent}{os.pathsep}{env.get('PATH', '')}"
    lib_paths = [str(path) for path in venv_library_paths() if path.exists()]
    if lib_paths:
        joined = os.pathsep.join(lib_paths)
        env["LD_LIBRARY_PATH"] = os.pathsep.join([joined, env.get("LD_LIBRARY_PATH", "")])
        env["LIBRARY_PATH"] = os.pathsep.join([joined, env.get("LIBRARY_PATH", "")])
    attempts: list[Attempt] = []
    for workload, mode, data_path, query_path, rows, cols, expected in WORKLOADS:
        for precision, dtype in PRECISIONS:
            rel = output_rel(precision, workload, rows, cols)
            attempts.append(
                Attempt(
                    machine=MACHINE,
                    method=METHOD,
                    precision=precision,
                    workload=workload,
                    scale=SCALE,
                    command=command_for(mode, data_path, query_path, rows, cols, dtype, RUN_ROOT / rel),
                    cwd=CUML_ROOT,
                    output_rel=rel,
                    log_rel=Path("logs") / METHOD_TOKEN / f"{rel.stem}.log",
                    timeout_seconds=TIMEOUT_SECONDS,
                    expected_value_count=expected,
                    env=env,
                )
            )
    return attempts


def main() -> int:
    fragment = RUN_ROOT / "fragments" / f"{METHOD_TOKEN}.csv"
    run_attempts(planned_attempts(), RUN_ROOT, fragment)
    print(f"[STAGE0P1] manifest_fragment={fragment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
