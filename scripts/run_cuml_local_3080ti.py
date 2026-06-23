#!/usr/bin/env python3
"""Stage 0 local cuML-KDE runner.

The script writes temporary evidence only. Accepted outputs are promoted to
persistent storage manually after review.
"""

from __future__ import annotations

import csv
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


CUML_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/baselines/cuml")
HELPER = CUML_ROOT / "scripts/cuml_exact_kde.py"
PYTHON_BIN = Path(
    "/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/venvs/cuml-kde/bin/python"
)
CUDA_HOME = Path("/usr/local/cuda-12.6")

DATA_ROOT = Path("/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation")
TMP_RESULT_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/tmp-results/stage0/cuml")
PERSISTENT_RESULT_ROOT = Path("/home/ubuntu/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation/exact/experiments/stage0")
EXACT_REPO_ROOT = Path("/home/ubuntu/Documents/workspace/GPU-accelerated_Kernel_Density_Exact/GPU-kernel-density-exact")
BASIC_GROUND_TRUTH_ROOT = PERSISTENT_RESULT_ROOT / "local-3080ti/basic/basic-scan-ground-truth"

MACHINE = "local-3080ti"
METHOD = "cuML-KDE"
METHOD_TOKEN = "cuml-kde"
ENGINE = "cuml"
BACKEND = "GPU"
PRECISION = "FP64"
DTYPE = "float64"
EXPECTED_TIMING_SCOPE = "score_samples_only"
SVM_SCOTT_B = "0.1"
KDV_SCOTT_B = "1"
SMOKE_SCOTT_B = "1"
KDV_ROWS = 1920
KDV_COLS = 2560
SMOKE_KDV_ROWS = 256
SMOKE_KDV_COLS = 256
SVM_BATCH_SIZE = 4096
KDV_BATCH_SIZE = 8192
PAIR_BUFFER_BYTES = 4 * 1024 * 1024 * 1024
ABS_TOLERANCE = 1e-3
REL_TOLERANCE = 1e-5
SMOKE_TIMEOUT_SECONDS = 600
SVM_TIMEOUT_SECONDS = 3600
KDV_TIMEOUT_SECONDS = 3600
GPU_QUERY_COMMAND = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version,compute_cap",
    "--format=csv,noheader",
]

SUSY_X = DATA_ROOT / "susy/SUSY_X.data"
SUSY_Q = DATA_ROOT / "susy/SUSY_qSet.data"
HOME_X = DATA_ROOT / "home/HT_Sensor_dataset_X.data"
HOME_Q = DATA_ROOT / "home/HT_Sensor_dataset_qSet.data"
MINIBOONE_X = DATA_ROOT / "miniboone/MiniBooNE_X.data"
MINIBOONE_Q = DATA_ROOT / "miniboone/MiniBooNE_qSet.data"
HOME_VIS_X = DATA_ROOT / "home_visualization/HT_Sensor_dataset_vis_X.data"
SUSY_VIS_X = DATA_ROOT / "susy_visualization/SUSY_vis_X.data"

SMOKE_SUSY_X = EXACT_REPO_ROOT / "data/SUSY_10000_X.data"
SMOKE_SUSY_Q = EXACT_REPO_ROOT / "data/SUSY_100_qSet.data"
SMOKE_SUSY_VIS_X = EXACT_REPO_ROOT / "data/SUSY_vis_10000_X.data"
SMOKE_SVM_REF = EXACT_REPO_ROOT / "results/SUSY_10000x100_basic_scan_fp64_scott_diag_b1.out"
SMOKE_KDV_REF = EXACT_REPO_ROOT / "results/SUSY_vis_10000_kdv_256x256_basic_scan_fp64_scott_diag_b1.out"


@dataclass(frozen=True)
class Workload:
    run_group: str
    workload: str
    dataset: str
    mode: str
    data_path: Path
    query_path: Path | None
    grid_rows: int | None
    grid_cols: int | None
    scale_mode: str
    scale_value: str


@dataclass
class RunRecord:
    workload: Workload
    output_path: Path
    log_path: Path
    command: list[str]
    return_code: str
    timeout_seconds: int
    cli_wall_seconds: float
    timing_scope: str
    execution_seconds: str
    query_count_from_binary: str
    qps: str
    status: str


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_HOME"] = str(CUDA_HOME)
    env["PATH"] = f"{PYTHON_BIN.parent}{os.pathsep}{CUDA_HOME / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{CUDA_HOME / 'lib64'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def shell_join(command: Iterable[str]) -> str:
    return shlex.join([str(part) for part in command])


def run_shell_capture(command: list[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=CUML_ROOT,
        env=env if env is not None else run_env(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.stdout.rstrip()


def matrix_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
    if len(first) < 2:
        raise ValueError(f"Invalid matrix header: {path}")
    return int(first[0]), int(first[1])


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def parse_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    for key in ("timing_scope", "execution_seconds", "query_count", "qps"):
        match = re.search(rf"^{key}:\s*(.+?)\s*$", log_text, re.MULTILINE)
        if match:
            metrics[key] = match.group(1)
    return metrics


def iter_floats(path: Path) -> Iterable[float]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            for token in line.split():
                yield float(token)


def compare_numeric_files(output_path: Path, reference_path: Path) -> dict[str, str]:
    max_abs = 0.0
    max_rel = 0.0
    failure_count = 0
    count = 0
    output_iter = iter_floats(output_path)
    reference_iter = iter_floats(reference_path)

    while True:
        sentinel = object()
        output_value = next(output_iter, sentinel)
        reference_value = next(reference_iter, sentinel)
        if output_value is sentinel and reference_value is sentinel:
            break
        if output_value is sentinel or reference_value is sentinel:
            failure_count += 1
            break

        count += 1
        diff = abs(float(output_value) - float(reference_value))
        max_abs = max(max_abs, diff)
        if reference_value != 0.0:
            max_rel = max(max_rel, diff / abs(float(reference_value)))
        threshold = max(ABS_TOLERANCE, REL_TOLERANCE * abs(float(reference_value)))
        if diff > threshold:
            failure_count += 1

    return {
        "value_count": str(count),
        "max_abs_err": f"{max_abs:.12g}",
        "max_rel_err": f"{max_rel:.12g}",
        "failure_count": str(failure_count),
        "tolerance": f"abs={ABS_TOLERANCE:g};rel={REL_TOLERANCE:g}",
        "status": "ok" if failure_count == 0 else "failed_tolerance",
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_command(commands_file: Path, command: list[str], timeout_seconds: int | None = None) -> None:
    with commands_file.open("a", encoding="utf-8") as handle:
        handle.write(f"cd {shlex.quote(str(CUML_ROOT))}\n")
        if timeout_seconds is not None:
            handle.write(f"# timeout_seconds: {timeout_seconds}\n")
        handle.write(shell_join(command))
        handle.write("\n\n")


def timeout_output_to_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    command: list[str],
    log_path: Path,
    commands_file: Path,
    timeout_seconds: int,
) -> tuple[str, float, str, bool]:
    append_command(commands_file, command, timeout_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=CUML_ROOT,
            env=run_env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.perf_counter() - start
        partial_output = timeout_output_to_text(exc.stdout or exc.output)
        log_text = (
            partial_output
            + f"\n[TIMEOUT] Command exceeded timeout_seconds={timeout_seconds}.\n"
            + f"[TIMEOUT] cli_wall_seconds={wall_seconds:.9f}\n"
        )
        write_text(log_path, log_text)
        return "timeout", wall_seconds, log_text, True

    wall_seconds = time.perf_counter() - start
    write_text(log_path, completed.stdout)
    return str(completed.returncode), wall_seconds, completed.stdout, False


def ensure_python_deps() -> None:
    check = [
        str(PYTHON_BIN),
        "-c",
        "import cupy; import cuml; from cuml.neighbors import KernelDensity",
    ]
    completed = subprocess.run(
        check,
        cwd=CUML_ROOT,
        env=run_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("Missing cuML Python environment:\n" + completed.stdout)


def run_inventory_command(label: str, command: list[str]) -> str:
    output = run_shell_capture(command)
    return f"{label}:\n{output}\n"


def collect_gpu_summary() -> dict[str, str]:
    output = run_shell_capture(GPU_QUERY_COMMAND)
    first_line = output.splitlines()[0] if output else ""
    parts = [part.strip() for part in first_line.split(",")]
    while len(parts) < 4:
        parts.append("")
    return {
        "gpu_name": parts[0],
        "gpu_memory_total": parts[1],
        "gpu_driver_version": parts[2],
        "gpu_compute_cap": parts[3],
    }


def write_inventory(path: Path, timestamp: str, git_branch: str, git_commit: str) -> None:
    content = [
        f"timestamp: {timestamp}",
        f"machine: {MACHINE}",
        f"hostname: {socket.gethostname()}",
        f"cuml_root: {CUML_ROOT}",
        f"helper: {HELPER}",
        f"python_bin: {PYTHON_BIN}",
        f"data_root: {DATA_ROOT}",
        f"tmp_result_root: {TMP_RESULT_ROOT}",
        f"persistent_result_root: {PERSISTENT_RESULT_ROOT}",
        f"basic_ground_truth_root: {BASIC_GROUND_TRUTH_ROOT}",
        f"method: {METHOD}",
        f"engine: {ENGINE}",
        f"backend: {BACKEND}",
        f"dtype: {DTYPE}",
        f"precision: {PRECISION}",
        f"timing_scope: {EXPECTED_TIMING_SCOPE}",
        f"svm_scott_b: {SVM_SCOTT_B}",
        f"kdv_scott_b: {KDV_SCOTT_B}",
        f"svm_batch_size: {SVM_BATCH_SIZE}",
        f"kdv_batch_size: {KDV_BATCH_SIZE}",
        f"pair_buffer_bytes: {PAIR_BUFFER_BYTES}",
        f"git_branch: {git_branch}",
        f"git_commit: {git_commit}",
        "",
        run_inventory_command("git_status", ["git", "status", "--short"]),
        run_inventory_command("uname", ["uname", "-a"]),
        run_inventory_command("lscpu", ["bash", "-lc", "lscpu | sed -n '1,40p'"]),
        run_inventory_command("nvidia-smi", GPU_QUERY_COMMAND),
        run_inventory_command("nvcc", ["nvcc", "--version"]),
        run_inventory_command("gcc", ["bash", "-lc", "gcc --version | head -1"]),
        run_inventory_command("g++", ["bash", "-lc", "g++ --version | head -1"]),
        run_inventory_command("cmake", ["bash", "-lc", "cmake --version | head -1"]),
        run_inventory_command("python", [str(PYTHON_BIN), "--version"]),
        run_inventory_command("cuml_cupy", [str(PYTHON_BIN), "-c", "import cuml, cupy; print('cuml', cuml.__version__); print('cupy', cupy.__version__)"]),
        run_inventory_command("pip_show", [str(PYTHON_BIN), "-m", "pip", "show", "cuml-cu12", "cupy-cuda12x"]),
    ]
    write_text(path, "\n".join(content))


def workload_query_count(workload: Workload) -> int:
    if workload.mode == "svm":
        assert workload.query_path is not None
        rows, _ = matrix_shape(workload.query_path)
        return rows
    assert workload.grid_rows is not None and workload.grid_cols is not None
    return workload.grid_rows * workload.grid_cols


def timeout_for_workload(workload: Workload) -> int:
    if workload.run_group == "smoke":
        return SMOKE_TIMEOUT_SECONDS
    if workload.mode == "svm":
        return SVM_TIMEOUT_SECONDS
    return KDV_TIMEOUT_SECONDS


def batch_size_for_workload(workload: Workload) -> int:
    data_rows, _ = matrix_shape(workload.data_path)
    cap = SVM_BATCH_SIZE if workload.mode == "svm" else KDV_BATCH_SIZE
    memory_limited = max(1, PAIR_BUFFER_BYTES // (data_rows * 8))
    return int(min(cap, memory_limited))


def workload_command(workload: Workload, output_path: Path) -> list[str]:
    batch_size = str(batch_size_for_workload(workload))
    base = [str(PYTHON_BIN), str(HELPER), workload.mode]
    if workload.mode == "svm":
        assert workload.query_path is not None
        base.extend([str(workload.query_path), str(workload.data_path), str(output_path)])
    else:
        assert workload.grid_rows is not None and workload.grid_cols is not None
        base.extend([
            "--rows",
            str(workload.grid_rows),
            "--cols",
            str(workload.grid_cols),
            str(workload.data_path),
            str(output_path),
        ])
    base.extend([
        "--scott-diag",
        workload.scale_value,
        "--batch-size",
        batch_size,
        "--dtype",
        DTYPE,
    ])
    return base


def output_name(workload: Workload) -> str:
    safe_b = workload.scale_value.replace(".", "p")
    if workload.mode == "kdv":
        assert workload.grid_rows is not None and workload.grid_cols is not None
        return (
            f"{workload.workload}_cuml-kde_fp64_"
            f"{workload.grid_rows}x{workload.grid_cols}_scott_diag_b{safe_b}.out"
        )
    return f"{workload.workload}_cuml-kde_fp64_scott_diag_b{safe_b}.out"


def run_workload(workload: Workload, run_dir: Path, commands_file: Path) -> RunRecord:
    output_path = run_dir / "outputs" / workload.run_group / output_name(workload)
    log_path = run_dir / "logs" / workload.run_group / f"{output_path.stem}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = workload_command(workload, output_path)
    timeout_seconds = timeout_for_workload(workload)

    print(f"[RUN] {workload.run_group} {workload.workload} (timeout={timeout_seconds}s)", flush=True)
    return_code, wall_seconds, log_text, timed_out = run_command(
        command,
        log_path,
        commands_file,
        timeout_seconds,
    )
    metrics = parse_metrics(log_text)

    status = "timeout" if timed_out else ("ok" if return_code == "0" else f"failed({return_code})")
    if status == "ok" and metrics.get("timing_scope") != EXPECTED_TIMING_SCOPE:
        status = "missing_or_bad_timing"
    if status == "ok" and metrics.get("query_count") != str(workload_query_count(workload)):
        status = "bad_query_count"

    return RunRecord(
        workload=workload,
        output_path=output_path,
        log_path=log_path,
        command=command,
        return_code=return_code,
        timeout_seconds=timeout_seconds,
        cli_wall_seconds=wall_seconds,
        timing_scope=metrics.get("timing_scope", ""),
        execution_seconds=metrics.get("execution_seconds", ""),
        query_count_from_binary=metrics.get("query_count", ""),
        qps=metrics.get("qps", ""),
        status=status,
    )


def summary_row(
    record: RunRecord,
    timestamp: str,
    git_branch: str,
    git_commit: str,
    gpu_info: dict[str, str],
) -> dict[str, str]:
    data_rows, dimension = matrix_shape(record.workload.data_path)
    query_rows = workload_query_count(record.workload)
    query_path = str(record.workload.query_path) if record.workload.query_path else ""
    grid_rows = str(record.workload.grid_rows) if record.workload.grid_rows is not None else ""
    grid_cols = str(record.workload.grid_cols) if record.workload.grid_cols is not None else ""
    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "hostname": socket.gethostname(),
        "run_group": record.workload.run_group,
        "method": METHOD,
        "method_token": METHOD_TOKEN,
        "engine": ENGINE,
        "backend": BACKEND,
        "precision": PRECISION,
        "dtype": DTYPE,
        "workload": record.workload.workload,
        "dataset": record.workload.dataset,
        "mode": record.workload.mode,
        "data_path": str(record.workload.data_path),
        "query_path": query_path,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "data_rows": str(data_rows),
        "query_rows": str(query_rows),
        "binary_query_count": record.query_count_from_binary,
        "dimension": str(dimension),
        "weights": "none",
        "scale_mode": record.workload.scale_mode,
        "scale_value": record.workload.scale_value,
        "scale": f"{record.workload.scale_mode} b={record.workload.scale_value}",
        "batch_size": str(batch_size_for_workload(record.workload)),
        "timing_scope": record.timing_scope,
        "runtime_s": record.execution_seconds,
        "qps": record.qps,
        "timeout_seconds": str(record.timeout_seconds),
        "cli_wall_seconds": f"{record.cli_wall_seconds:.9f}",
        "output_path": str(record.output_path),
        "log_path": str(record.log_path),
        "command": shell_join(record.command),
        "status": record.status,
        "return_code": str(record.return_code),
        "gpu_name": gpu_info["gpu_name"],
        "gpu_memory_total": gpu_info["gpu_memory_total"],
        "gpu_driver_version": gpu_info["gpu_driver_version"],
        "gpu_compute_cap": gpu_info["gpu_compute_cap"],
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def correctness_row(
    timestamp: str,
    git_branch: str,
    git_commit: str,
    workload: Workload,
    reference_method: str,
    reference_output: Path,
    output_path: Path,
) -> dict[str, str]:
    if not output_path.is_file() or not reference_output.is_file():
        metrics = {
            "value_count": "",
            "max_abs_err": "",
            "max_rel_err": "",
            "failure_count": "",
            "tolerance": f"abs={ABS_TOLERANCE:g};rel={REL_TOLERANCE:g}",
            "status": "missing_output",
        }
    else:
        metrics = compare_numeric_files(output_path, reference_output)

    return {
        "timestamp": timestamp,
        "machine": MACHINE,
        "run_group": workload.run_group,
        "workload": workload.workload,
        "mode": workload.mode,
        "scale": f"{workload.scale_mode} b={workload.scale_value}",
        "reference_method": reference_method,
        "method": METHOD,
        "precision": PRECISION,
        "reference_output": str(reference_output),
        "output_path": str(output_path),
        "value_count": metrics["value_count"],
        "max_abs_err": metrics["max_abs_err"],
        "max_rel_err": metrics["max_rel_err"],
        "failure_count": metrics["failure_count"],
        "tolerance": metrics["tolerance"],
        "status": metrics["status"],
        "git_branch": git_branch,
        "git_commit": git_commit,
    }


def full_reference_for(workload: Workload) -> Path:
    safe_b = workload.scale_value.replace(".", "p")
    if workload.mode == "kdv":
        assert workload.grid_rows is not None and workload.grid_cols is not None
        return (
            BASIC_GROUND_TRUTH_ROOT
            / f"{workload.workload}_basic-scan_fp64_{workload.grid_rows}x{workload.grid_cols}_scott_diag_b{safe_b}.out"
        )
    return BASIC_GROUND_TRUTH_ROOT / f"{workload.workload}_basic-scan_fp64_scott_diag_b{safe_b}.out"


def main() -> int:
    if len(sys.argv) > 1:
        if sys.argv[1:] in (["-h"], ["--help"]):
            print(
                "usage: run_cuml_local_3080ti.py\n\n"
                "Runs the Stage 0 local cuML-KDE smoke and full workloads.\n"
                "Outputs are written under tmp-results/stage0/cuml/ and are\n"
                "not promoted to persistent storage automatically."
            )
            return 0
        print("run_cuml_local_3080ti.py does not accept arguments.", file=sys.stderr)
        return 2

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = TMP_RESULT_ROOT / f"{MACHINE}_{timestamp}"
    commands_file = run_dir / "commands.sh"
    inventory_file = run_dir / "machine_inventory.txt"
    summary_file = run_dir / "summary.csv"
    correctness_file = run_dir / "correctness.csv"

    run_dir.mkdir(parents=True, exist_ok=True)
    write_text(
        commands_file,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        f"# Generated by {Path(__file__).name} at {timestamp}.\n\n",
    )
    commands_file.chmod(0o755)

    required_files = [
        HELPER,
        PYTHON_BIN,
        SUSY_X,
        SUSY_Q,
        HOME_X,
        HOME_Q,
        MINIBOONE_X,
        MINIBOONE_Q,
        HOME_VIS_X,
        SUSY_VIS_X,
        SMOKE_SUSY_X,
        SMOKE_SUSY_Q,
        SMOKE_SUSY_VIS_X,
        SMOKE_SVM_REF,
        SMOKE_KDV_REF,
        BASIC_GROUND_TRUTH_ROOT / "svm_susy_basic-scan_fp64_scott_diag_b0p1.out",
        BASIC_GROUND_TRUTH_ROOT / "svm_home_basic-scan_fp64_scott_diag_b0p1.out",
        BASIC_GROUND_TRUTH_ROOT / "svm_miniboone_basic-scan_fp64_scott_diag_b0p1.out",
        BASIC_GROUND_TRUTH_ROOT / "kdv_home_visual_basic-scan_fp64_1920x2560_scott_diag_b1.out",
        BASIC_GROUND_TRUTH_ROOT / "kdv_susy_visual_basic-scan_fp64_1920x2560_scott_diag_b1.out",
    ]
    require_files(required_files)

    git_branch = run_shell_capture(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_commit = run_shell_capture(["git", "rev-parse", "HEAD"])
    gpu_info = collect_gpu_summary()
    ensure_python_deps()
    write_inventory(inventory_file, timestamp, git_branch, git_commit)

    smoke_workloads = [
        Workload("smoke", "smoke_svm_susy_10000x100", "SUSY validation subset", "svm", SMOKE_SUSY_X, SMOKE_SUSY_Q, None, None, "scott_diag", SMOKE_SCOTT_B),
        Workload("smoke", "smoke_kdv_susy_vis_10000_256x256", "SUSY visualization validation subset", "kdv", SMOKE_SUSY_VIS_X, None, SMOKE_KDV_ROWS, SMOKE_KDV_COLS, "scott_diag", SMOKE_SCOTT_B),
    ]
    full_workloads = [
        Workload("full", "svm_susy", "SUSY", "svm", SUSY_X, SUSY_Q, None, None, "scott_diag", SVM_SCOTT_B),
        Workload("full", "svm_home", "Home", "svm", HOME_X, HOME_Q, None, None, "scott_diag", SVM_SCOTT_B),
        Workload("full", "svm_miniboone", "MiniBooNE", "svm", MINIBOONE_X, MINIBOONE_Q, None, None, "scott_diag", SVM_SCOTT_B),
        Workload("full", "kdv_home_visual", "Home visualization", "kdv", HOME_VIS_X, None, KDV_ROWS, KDV_COLS, "scott_diag", KDV_SCOTT_B),
        Workload("full", "kdv_susy_visual", "SUSY visualization", "kdv", SUSY_VIS_X, None, KDV_ROWS, KDV_COLS, "scott_diag", KDV_SCOTT_B),
    ]

    records: list[RunRecord] = []
    for workload in [*smoke_workloads, *full_workloads]:
        records.append(run_workload(workload, run_dir, commands_file))

    summary_fields = [
        "timestamp",
        "machine",
        "hostname",
        "run_group",
        "method",
        "method_token",
        "engine",
        "backend",
        "precision",
        "dtype",
        "workload",
        "dataset",
        "mode",
        "data_path",
        "query_path",
        "grid_rows",
        "grid_cols",
        "data_rows",
        "query_rows",
        "binary_query_count",
        "dimension",
        "weights",
        "scale_mode",
        "scale_value",
        "scale",
        "batch_size",
        "timing_scope",
        "runtime_s",
        "qps",
        "timeout_seconds",
        "cli_wall_seconds",
        "output_path",
        "log_path",
        "command",
        "status",
        "return_code",
        "gpu_name",
        "gpu_memory_total",
        "gpu_driver_version",
        "gpu_compute_cap",
        "git_branch",
        "git_commit",
    ]
    write_csv(
        summary_file,
        [summary_row(r, timestamp, git_branch, git_commit, gpu_info) for r in records],
        summary_fields,
    )

    record_by_key = {(r.workload.run_group, r.workload.workload): r for r in records}
    correctness_rows: list[dict[str, str]] = []
    for workload in smoke_workloads:
        record = record_by_key[(workload.run_group, workload.workload)]
        reference = SMOKE_SVM_REF if workload.mode == "svm" else SMOKE_KDV_REF
        correctness_rows.append(
            correctness_row(
                timestamp,
                git_branch,
                git_commit,
                workload,
                "checked_results",
                reference,
                record.output_path,
            )
        )

    for workload in full_workloads:
        record = record_by_key[(workload.run_group, workload.workload)]
        correctness_rows.append(
            correctness_row(
                timestamp,
                git_branch,
                git_commit,
                workload,
                "Basic-Scan",
                full_reference_for(workload),
                record.output_path,
            )
        )

    correctness_fields = [
        "timestamp",
        "machine",
        "run_group",
        "workload",
        "mode",
        "scale",
        "reference_method",
        "method",
        "precision",
        "reference_output",
        "output_path",
        "value_count",
        "max_abs_err",
        "max_rel_err",
        "failure_count",
        "tolerance",
        "status",
        "git_branch",
        "git_commit",
    ]
    write_csv(correctness_file, correctness_rows, correctness_fields)

    run_failures = [record for record in records if record.status != "ok"]
    correctness_failures = [row for row in correctness_rows if row["status"] != "ok"]

    print(f"[DONE] run_dir={run_dir}")
    print(f"[DONE] summary={summary_file}")
    print(f"[DONE] correctness={correctness_file}")
    print("[DONE] Persistent promotion is manual after review.")

    if run_failures or correctness_failures:
        print(
            f"[ERROR] run_failures={len(run_failures)} correctness_failures={len(correctness_failures)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
