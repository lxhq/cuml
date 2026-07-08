#!/usr/bin/env python3
"""Stage 2 remote H100 runner for cuML-KDE 26.06 exact KDE."""

from __future__ import annotations

import csv
import hashlib
import math
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
from typing import Iterable, TextIO


MACHINE = "remote-h100"
METHOD = "cuML-KDE 26.06"
METHOD_TOKEN = "cuml-kde-2606"
WORKSPACE_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact")
CUML_ROOT = WORKSPACE_ROOT / "baselines/cuml"
HELPER = CUML_ROOT / "scripts/cuml_stage1p1_exact_kde.py"
PYTHON_BIN = WORKSPACE_ROOT / "venvs/kde-baselines-cuml2606/bin/python"
CUDA_HOME = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.4"))
TMP_ROOT = WORKSPACE_ROOT / "tmp-results/stage2"
PREPARED_ROOT = TMP_ROOT / "prepared"
GROUND_TRUTH_ROOT = TMP_ROOT / "ground_truth"
RUN_ROOT = TMP_ROOT / "runs/cuml/remote-h100"
TIMEOUT_SECONDS = 24 * 60 * 60
SCALE_ARGS = ("--scott-diag", "1")
SCALE = "scott_diag b=1"
PRECISION_LABEL = "FP64"
DTYPE = "float64"
TIMING_MODES = ("cold_start", "warm")
EXPECTED_TIMING_SCOPE = "end_to_end_in_memory"
WARMUP_POLICIES = {
    "cold_start": "none",
    "warm": "full_untimed_cuml_pipeline",
}


@dataclass(frozen=True)
class Workload:
    name: str
    data_rows: int
    query_rows: int
    dim: int

    @property
    def data_path(self) -> Path:
        return PREPARED_ROOT / self.name / f"{self.name}_X.data.zst"

    @property
    def query_path(self) -> Path:
        return PREPARED_ROOT / self.name / f"{self.name}_qSet.data.zst"

    @property
    def reference_path(self) -> Path:
        return GROUND_TRUTH_ROOT / f"{self.name}_basic-scan_fp64_scott_b1.out.zst"


WORKLOADS = (
    Workload("stage2_higgs_100k", 10_500_000, 100_000, 28),
    Workload("stage2_hepmass_100k", 7_000_000, 100_000, 28),
    Workload("stage2_epsilon_2000d", 400_000, 100_000, 2000),
)


def shell_join(command: Iterable[object]) -> str:
    return shlex.join([str(part) for part in command])


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))


def run_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["CUDA_HOME"] = str(CUDA_HOME)
    env["PATH"] = os.pathsep.join([str(PYTHON_BIN.parent), str(CUDA_HOME / "bin"), env.get("PATH", "")])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(PYTHON_BIN.parents[1] / "lib"), str(CUDA_HOME / "lib64"), env.get("LD_LIBRARY_PATH", "")]
    )
    return env


def append_command(commands_file: Path, cwd: Path, command: list[object], timeout_seconds: int) -> None:
    with commands_file.open("a", encoding="utf-8") as handle:
        handle.write(f"cd {shlex.quote(str(cwd))}\n")
        handle.write(f"# timeout_seconds: {timeout_seconds}\n")
        handle.write(shell_join(command))
        handle.write("\n\n")


def timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_command(
    command: list[object],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    commands_file: Path,
    timeout_seconds: int,
) -> tuple[str, float, str, bool]:
    append_command(commands_file, cwd, command, timeout_seconds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        wall_seconds = time.perf_counter() - start
        log_text = (
            timeout_output(exc.stdout or exc.output)
            + f"\n[TIMEOUT] timeout_seconds={timeout_seconds}\n"
            + f"[TIMEOUT] full_command_wall_seconds={wall_seconds:.9f}\n"
        )
        log_path.write_text(log_text, encoding="utf-8")
        return "timeout", wall_seconds, log_text, True

    wall_seconds = time.perf_counter() - start
    log_text = (
        f"$ {shell_join(command)}\n"
        f"return_code: {completed.returncode}\n"
        f"full_command_wall_seconds: {wall_seconds:.9f}\n\n"
        f"{completed.stdout or ''}"
    )
    log_path.write_text(log_text, encoding="utf-8")
    return str(completed.returncode), wall_seconds, log_text, False


def parse_metrics(log_text: str) -> dict[str, str]:
    metrics: dict[str, str] = {}
    patterns = {
        "cuml_package": r"^cuML package:\s*(.+?)\s*$",
        "timing_scope": r"^timing_scope:\s*(.+?)\s*$",
        "execution_seconds": r"^execution_seconds:\s*(.+?)\s*$",
        "query_count": r"^query_count:\s*(.+?)\s*$",
        "qps": r"^qps:\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, log_text, re.MULTILINE)
        if match:
            metrics[key] = match.group(1)
    return metrics


def open_vector(path: Path) -> tuple[TextIO, subprocess.Popen[str] | None]:
    if path.name.endswith(".zst"):
        proc = subprocess.Popen(
            ["zstd", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.stdout is None:
            raise RuntimeError(f"failed to open zstd stream for {path}")
        return proc.stdout, proc
    return path.open("r", encoding="utf-8"), None


def close_vector(handle: TextIO, proc: subprocess.Popen[str] | None) -> None:
    handle.close()
    if proc is None:
        return
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.strip() or f"zstd failed with code {proc.returncode}")


def iter_floats(path: Path) -> Iterable[float]:
    handle, proc = open_vector(path)
    try:
        for line in handle:
            for token in line.split():
                yield float(token)
    finally:
        close_vector(handle, proc)


def compare_output(output_path: Path, reference_path: Path) -> dict[str, str]:
    if not output_path.is_file():
        return {"correctness_status": "missing_output"}
    if not reference_path.is_file():
        return {"correctness_status": "missing_reference"}

    out_iter = iter(iter_floats(output_path))
    ref_iter = iter(iter_floats(reference_path))
    sentinel = object()
    count = 0
    max_abs = 0.0
    sum_abs = 0.0
    max_rel = 0.0
    sum_rel = 0.0
    zero_reference = 0
    nonzero_output_zero_reference = 0
    while True:
        out = next(out_iter, sentinel)
        ref = next(ref_iter, sentinel)
        if out is sentinel and ref is sentinel:
            break
        if out is sentinel or ref is sentinel:
            return {"correctness_status": "count_mismatch", "output_count": str(count)}
        out = float(out)
        ref = float(ref)
        diff = abs(out - ref)
        if abs(ref) == 0.0:
            zero_reference += 1
            rel = 0.0 if diff == 0.0 else math.inf
            if diff != 0.0:
                nonzero_output_zero_reference += 1
        else:
            rel = diff / abs(ref)
        max_abs = max(max_abs, diff)
        sum_abs += diff
        max_rel = max(max_rel, rel)
        sum_rel += rel
        count += 1

    if count == 0:
        return {"correctness_status": "empty_output"}
    return {
        "correctness_status": "compared",
        "output_count": str(count),
        "max_abs_err": f"{max_abs:.17g}",
        "mean_abs_err": f"{sum_abs / count:.17g}",
        "max_rel_err": "inf" if math.isinf(max_rel) else f"{max_rel:.17g}",
        "mean_rel_err": "inf" if math.isinf(sum_rel) else f"{sum_rel / count:.17g}",
        "zero_reference": str(zero_reference),
        "nonzero_output_zero_reference": str(nonzero_output_zero_reference),
    }


def git_value(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=CUML_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def write_inventory(path: Path, env: dict[str, str]) -> None:
    commands = [
        ["hostname"],
        ["date", "-Is"],
        ["nvidia-smi"],
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv"],
        [str(PYTHON_BIN), "--version"],
        [str(PYTHON_BIN), "-c", "import cupy, cuml, zstandard; print('cupy', cupy.__version__); print('cuml', cuml.__version__)"],
        ["git", "-C", str(CUML_ROOT), "status", "--short", "--branch"],
        ["git", "-C", str(CUML_ROOT), "rev-parse", "HEAD"],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"machine: {MACHINE}\n")
        handle.write(f"hostname: {socket.gethostname()}\n")
        handle.write(f"cuml_root: {CUML_ROOT}\n")
        handle.write(f"python_bin: {PYTHON_BIN}\n")
        handle.write(f"tmp_root: {TMP_ROOT}\n")
        handle.write(f"prepared_root: {PREPARED_ROOT}\n")
        handle.write(f"ground_truth_root: {GROUND_TRUTH_ROOT}\n")
        handle.write(f"timeout_seconds: {TIMEOUT_SECONDS}\n\n")
        for command in commands:
            handle.write(f"$ {shell_join(command)}\n")
            completed = subprocess.run(
                command,
                cwd=CUML_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(completed.stdout)
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256sums(run_dir: Path) -> None:
    checksum_path = run_dir / "SHA256SUMS"
    rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != checksum_path.name:
            rows.append(f"{sha256_file(path)}  ./{path.relative_to(run_dir)}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def command_for(
    workload: Workload,
    timing_mode: str,
    output_path: Path,
    adapter_summary: Path,
    source_commit: str,
) -> list[object]:
    return [
        PYTHON_BIN,
        HELPER,
        "svm",
        workload.data_path,
        workload.query_path,
        output_path,
        *SCALE_ARGS,
        "--dtype",
        DTYPE,
        "--timing-mode",
        timing_mode,
        "--machine",
        MACHINE,
        "--workload",
        workload.name,
        "--source-commit",
        source_commit,
        "--summary-csv",
        adapter_summary,
    ]


def run_status(
    return_code: str,
    timed_out: bool,
    metrics: dict[str, str],
    workload: Workload,
) -> str:
    if timed_out:
        return "timeout"
    if return_code != "0":
        return f"failed({return_code})"
    if metrics.get("timing_scope") != EXPECTED_TIMING_SCOPE:
        return "bad_timing_scope"
    if metrics.get("query_count") != str(workload.query_rows):
        return "bad_query_count"
    package = metrics.get("cuml_package", "")
    if not package.startswith("26.06"):
        return f"bad_cuml_version({package})"
    return "ok"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "machine",
        "workload",
        "method",
        "cuml_package",
        "precision",
        "input_format",
        "timing_mode",
        "warmup_policy",
        "timing_scope",
        "runtime_s",
        "qps",
        "full_command_wall_s",
        "correctness_status",
        "output_count",
        "max_abs_err",
        "mean_abs_err",
        "max_rel_err",
        "mean_rel_err",
        "zero_reference",
        "nonzero_output_zero_reference",
        "return_code",
        "timed_out",
        "timeout_seconds",
        "data_path",
        "query_path",
        "reference_path",
        "output_path",
        "log_path",
        "command",
        "git_branch",
        "git_commit",
        "status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    require_paths([CUML_ROOT, HELPER, PYTHON_BIN, PREPARED_ROOT, GROUND_TRUTH_ROOT, Path("/usr/bin/zstd")])
    for workload in WORKLOADS:
        require_paths([workload.data_path, workload.query_path, workload.reference_path])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / timestamp
    log_dir = run_dir / "logs"
    output_dir = run_dir / "outputs"
    commands_file = run_dir / "commands.sh"
    summary_file = run_dir / "summary.csv"
    adapter_summary = run_dir / "cuml_adapter_summary.csv"
    run_dir.mkdir(parents=True, exist_ok=False)
    commands_file.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n", encoding="utf-8")
    commands_file.chmod(0o755)

    env = run_env()
    write_inventory(run_dir / "machine_inventory.txt", env)

    branch = git_value(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = git_value(["rev-parse", "HEAD"])
    rows: list[dict[str, str]] = []
    for workload in WORKLOADS:
        for timing_mode in TIMING_MODES:
            output_path = (
                output_dir
                / workload.name
                / timing_mode
                / f"{workload.name}_{METHOD_TOKEN}_{DTYPE}_{timing_mode}_scott_b1.out"
            )
            log_path = log_dir / workload.name / f"{workload.name}_{METHOD_TOKEN}_{DTYPE}_{timing_mode}.log"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            command = command_for(workload, timing_mode, output_path, adapter_summary, commit)
            print(f"[RUN] {workload.name} {METHOD} {timing_mode}", flush=True)
            return_code, wall_seconds, log_text, timed_out = run_command(
                command,
                cwd=CUML_ROOT,
                env=env,
                log_path=log_path,
                commands_file=commands_file,
                timeout_seconds=TIMEOUT_SECONDS,
            )
            metrics = parse_metrics(log_text)
            status = run_status(return_code, timed_out, metrics, workload)
            correctness = compare_output(output_path, workload.reference_path) if status == "ok" else {}
            rows.append(
                {
                    "machine": MACHINE,
                    "workload": workload.name,
                    "method": METHOD,
                    "cuml_package": metrics.get("cuml_package", ""),
                    "precision": PRECISION_LABEL,
                    "input_format": "dense .data.zst",
                    "timing_mode": timing_mode,
                    "warmup_policy": WARMUP_POLICIES[timing_mode],
                    "timing_scope": metrics.get("timing_scope", ""),
                    "runtime_s": metrics.get("execution_seconds", ""),
                    "qps": metrics.get("qps", ""),
                    "full_command_wall_s": f"{wall_seconds:.9f}",
                    "correctness_status": correctness.get("correctness_status", status),
                    "output_count": correctness.get("output_count", ""),
                    "max_abs_err": correctness.get("max_abs_err", ""),
                    "mean_abs_err": correctness.get("mean_abs_err", ""),
                    "max_rel_err": correctness.get("max_rel_err", ""),
                    "mean_rel_err": correctness.get("mean_rel_err", ""),
                    "zero_reference": correctness.get("zero_reference", ""),
                    "nonzero_output_zero_reference": correctness.get(
                        "nonzero_output_zero_reference", ""
                    ),
                    "return_code": return_code,
                    "timed_out": str(timed_out).lower(),
                    "timeout_seconds": str(TIMEOUT_SECONDS),
                    "data_path": str(workload.data_path),
                    "query_path": str(workload.query_path),
                    "reference_path": str(workload.reference_path),
                    "output_path": str(output_path),
                    "log_path": str(log_path),
                    "command": shell_join(command),
                    "git_branch": branch,
                    "git_commit": commit,
                    "status": status,
                }
            )
            write_csv(summary_file, rows)

    write_sha256sums(run_dir)
    print(f"[stage2-cuml] run_dir={run_dir}")
    print(f"[stage2-cuml] summary={summary_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[stage2-cuml][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
