#!/usr/bin/env python3
"""Run Stage 1-p1 cuML-KDE accepted SVM rows on the remote H100 machine."""

from __future__ import annotations

import csv
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


MACHINE = "remote-h100"
METHOD = "cuML-KDE"
TIMING_SCOPE = "end_to_end_in_memory"
SCOTT_B_VALUE = "1"
SCOTT_B_LABEL = "1"
TIMEOUT_SECONDS = 3600
CUML_SOURCE_COMMIT = "62cd497281319aa6d7aa5b5255a2624215059b41"

WORKSPACE_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact")
REPO_ROOT = WORKSPACE_ROOT / "baselines/cuml"
PYTHON = WORKSPACE_ROOT / "venvs/kde-baselines-cuml2606/bin/python"
ADAPTER = REPO_ROOT / "scripts/cuml_stage1p1_exact_kde.py"

DATA_ROOT = Path(
    "/home/lxheq/Documents/workspace/dataset/"
    "GPU-accelerated_Kernel_Density_Computation"
)
GROUND_TRUTH_ROOT = (
    DATA_ROOT / "exact/experiments/stage0/ground_truth"
)
TMP_ROOT = WORKSPACE_ROOT / "tmp-results/stage1-p1/cuml/remote-h100"


@dataclass(frozen=True)
class Workload:
    name: str
    data: Path
    query: Path


WORKLOADS = (
    Workload(
        "svm_susy",
        DATA_ROOT / "susy/SUSY_X.data",
        DATA_ROOT / "susy/SUSY_qSet.data",
    ),
    Workload(
        "svm_home",
        DATA_ROOT / "home/HT_Sensor_dataset_X.data",
        DATA_ROOT / "home/HT_Sensor_dataset_qSet.data",
    ),
    Workload(
        "svm_miniboone",
        DATA_ROOT / "miniboone/MiniBooNE_X.data",
        DATA_ROOT / "miniboone/MiniBooNE_qSet.data",
    ),
)

PRECISIONS = (
    ("FP64", "float64"),
    ("FP32", "float32"),
)

TIMING_MODES = (
    "cold_start",
    "warm",
)


def ground_truth_path(workload: Workload) -> Path:
    return (
        GROUND_TRUTH_ROOT
        / f"{workload.name}_basic-scan_fp64_scott_diag_b{SCOTT_B_LABEL}.out"
    )


def output_name(workload: Workload, precision_label: str, timing_mode: str) -> str:
    return (
        f"{MACHINE}_cuml-kde_{precision_label.lower()}_{timing_mode}_"
        f"{workload.name}_scott_diag_b{SCOTT_B_LABEL}.out"
    )


def require_existing_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing required paths:\n  {joined}")


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def append_csv(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "machine",
        "method",
        "workload",
        "precision",
        "timing_mode",
        "timing_scope",
        "return_code",
        "timed_out",
        "cli_wall_seconds",
        "log_path",
        "output_path",
        "status",
    ]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_inventory(run_dir: Path) -> None:
    commands = [
        ["hostname"],
        ["date", "-Is"],
        ["nvidia-smi"],
        [str(PYTHON), "--version"],
        [
            str(PYTHON),
            "-c",
            "import cuml, cupy; "
            "print('cuml', cuml.__version__); "
            "print('cupy', cupy.__version__)",
        ],
    ]

    inventory = run_dir / "machine_inventory.txt"
    with inventory.open("w", encoding="utf-8") as handle:
        for command in commands:
            handle.write(f"$ {shlex.join(command)}\n")
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=60,
                    check=False,
                )
                handle.write(completed.stdout)
                handle.write(f"\n[exit {completed.returncode}]\n\n")
            except Exception as exc:
                handle.write(f"[inventory command failed: {exc}]\n\n")


def run_attempt(
    command: list[str],
    log_path: Path,
    commands_path: Path,
) -> tuple[str, bool, float]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"

    with commands_path.open("a", encoding="utf-8") as handle:
        handle.write(shlex.join(command))
        handle.write("\n")

    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n\n")
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
            log.write(completed.stdout)
            log.write(f"\n[exit {completed.returncode}]\n")
            return str(completed.returncode), False, time.perf_counter() - started
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                log.write(exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode())
            log.write(f"\n[TIMEOUT after {TIMEOUT_SECONDS} seconds]\n")
            return "timeout", True, time.perf_counter() - started


def main() -> int:
    all_required = [PYTHON, ADAPTER]
    for workload in WORKLOADS:
        all_required.extend([workload.data, workload.query, ground_truth_path(workload)])
    require_existing_paths(all_required)

    run_dir = TMP_ROOT / run_id()
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "logs"
    outputs_dir.mkdir(parents=True, exist_ok=False)
    logs_dir.mkdir(parents=True, exist_ok=False)

    commands_path = run_dir / "commands.sh"
    commands_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
    summary_csv = run_dir / "summary.csv"
    runner_status_csv = run_dir / "runner_status.csv"

    write_inventory(run_dir)

    print(f"Run directory: {run_dir}")
    failures = 0
    for workload in WORKLOADS:
        reference = ground_truth_path(workload)
        for precision_label, dtype in PRECISIONS:
            for timing_mode in TIMING_MODES:
                output_path = outputs_dir / output_name(workload, precision_label, timing_mode)
                log_path = (
                    logs_dir
                    / output_name(workload, precision_label, timing_mode).replace(
                        ".out", ".log"
                    )
                )
                command = [
                    str(PYTHON),
                    str(ADAPTER),
                    "svm",
                    str(workload.data),
                    str(workload.query),
                    str(output_path),
                    "--scott-diag",
                    SCOTT_B_VALUE,
                    "--dtype",
                    dtype,
                    "--reference",
                    str(reference),
                    "--summary-csv",
                    str(summary_csv),
                    "--timing-mode",
                    timing_mode,
                    "--machine",
                    MACHINE,
                    "--workload",
                    workload.name,
                    "--source-commit",
                    CUML_SOURCE_COMMIT,
                ]

                print(
                    f"[run] {workload.name} {precision_label} {timing_mode} "
                    f"(timeout={TIMEOUT_SECONDS}s)"
                )
                return_code, timed_out, wall_seconds = run_attempt(
                    command, log_path, commands_path
                )
                status = "ok" if return_code == "0" else "timeout" if timed_out else "failed"
                if status != "ok":
                    failures += 1
                append_csv(
                    runner_status_csv,
                    {
                        "machine": MACHINE,
                        "method": METHOD,
                        "workload": workload.name,
                        "precision": precision_label,
                        "timing_mode": timing_mode,
                        "timing_scope": TIMING_SCOPE,
                        "return_code": return_code,
                        "timed_out": str(timed_out).lower(),
                        "cli_wall_seconds": f"{wall_seconds:.9f}",
                        "log_path": str(log_path),
                        "output_path": str(output_path),
                        "status": status,
                    },
                )
                print(f"[{status}] wall={wall_seconds:.3f}s log={log_path}")

    print(f"Summary CSV: {summary_csv}")
    print(f"Runner status CSV: {runner_status_csv}")
    print(f"Outputs: {outputs_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
