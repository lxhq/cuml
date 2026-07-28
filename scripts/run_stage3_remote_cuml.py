#!/usr/bin/env python3
"""Run resumable Stage 3 cuML FP64 exact-KDE baselines on the remote H100."""

from __future__ import annotations

import csv
import hashlib
import json
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
from typing import Iterable, Iterator, TextIO


MACHINE = "remote-h100"
METHOD = "cuML-KDE 26.06"
WORKSPACE_ROOT = Path("/home/lxheq/Documents/workspace/GPU-accelerated_Kernel_Density_Exact")
CUML_ROOT = WORKSPACE_ROOT / "baselines/cuml-stage3"
EXACT_STAGE3_ROOT = WORKSPACE_ROOT / "GPU-kernel-density-exact-stage3"
HELPER = CUML_ROOT / "scripts/cuml_stage1p1_exact_kde.py"
PYTHON_BIN = WORKSPACE_ROOT / "venvs/kde-baselines-cuml2606/bin/python"
CUDA_HOME = Path("/usr/local/cuda-12.4")
DATA_ROOT = Path(
    "/home/lxheq/Documents/workspace/dataset/GPU-accelerated_Kernel_Density_Computation"
)
GROUND_TRUTH_ROOT = DATA_ROOT / "exact/experiments/stage3/ground_truth"
GROUND_TRUTH_MANIFEST = GROUND_TRUTH_ROOT / "manifest.csv"
RUN_ROOT = WORKSPACE_ROOT / "tmp-results/stage3/baselines/cuml/remote-h100"
LOG_ROOT = RUN_ROOT / "logs"
RECORD_ROOT = RUN_ROOT / "records"
OUTPUT_ROOT = RUN_ROOT / "plain-outputs"
SUMMARY_PATH = RUN_ROOT / "summary.csv"
COMMANDS_PATH = RUN_ROOT / "commands.sh"
INVENTORY_PATH = RUN_ROOT / "machine_inventory.txt"
CHECKSUM_PATH = RUN_ROOT / "SHA256SUMS"

EXPECTED_BRANCH = "stage3"
EXPECTED_CUML_VERSION_PREFIX = "26.06"
EXPECTED_TIMING_SCOPE = "end_to_end_in_memory"
TIMING_MODE = "cold_start"
TIMEOUT_SECONDS = 24 * 60 * 60
SCALE = "scott_diag b=1"
SCALE_ARGS = ("--scott-diag", "1")
PRECISION = "FP64"
DTYPE = "float64"
RELATIVE_TOLERANCE = 1e-5


@dataclass(frozen=True)
class Workload:
    name: str
    data_rows: int
    query_rows: int
    dimensions: int

    @property
    def dataset_dir(self) -> Path:
        return DATA_ROOT / self.name

    @property
    def data_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_X.data.zst"

    @property
    def query_path(self) -> Path:
        return self.dataset_dir / f"{self.name}_qSet.data.zst"

    @property
    def reference_path(self) -> Path:
        return GROUND_TRUTH_ROOT / f"{self.name}_basic-scan_fp64_scott_b1.out.zst"

    @property
    def output_path(self) -> Path:
        return OUTPUT_ROOT / f"{self.name}_cuml_fp64_scott_b1.out"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / f"{self.name}.log"

    @property
    def record_path(self) -> Path:
        return RECORD_ROOT / f"{self.name}.json"


WORKLOADS = (
    Workload("vk_lsvd", 19_527_601, 100_000, 64),
    Workload("tencent_chinese_100", 12_187_936, 100_000, 100),
    Workload("tencent_english_200", 6_496_681, 100_000, 200),
    Workload("wolt_food_clip_512", 1_620_611, 100_000, 512),
    Workload("fd", 5_902_200, 100_000, 900),
    Workload("ocr", 4_070_000, 100_000, 1_156),
    Workload("epsilon", 400_000, 100_000, 2_000),
)


SUMMARY_FIELDS = [
    "machine",
    "workload",
    "method",
    "precision",
    "scale",
    "data_rows",
    "query_rows",
    "dimensions",
    "timing_mode",
    "timing_scope",
    "cold_execution_seconds",
    "qps",
    "correctness",
    "correctness_status",
    "failure_count",
    "output_count",
    "max_abs_err",
    "mean_abs_err",
    "max_rel_err",
    "mean_rel_err",
    "zero_reference_count",
    "nonzero_output_zero_reference_count",
    "full_command_wall_seconds",
    "cuml_version",
    "data_sha256",
    "query_sha256",
    "reference_sha256",
    "temporary_output_sha256",
    "git_branch",
    "git_commit",
    "main_goal_commit",
    "log_path",
    "command",
    "accepted_at",
    "status",
]


def shell_join(command: Iterable[object]) -> str:
    return shlex.join([str(part) for part in command])


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))


def git_value(repo: Path, arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or f"git failed in {repo}")
    return completed.stdout.strip()


def require_clean_stage3() -> tuple[str, str]:
    branch = git_value(CUML_ROOT, ["rev-parse", "--abbrev-ref", "HEAD"])
    commit = git_value(CUML_ROOT, ["rev-parse", "HEAD"])
    status = git_value(CUML_ROOT, ["status", "--porcelain", "--untracked-files=all"])
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH}, found {branch}.")
    if status:
        raise RuntimeError(
            "The cuML Stage 3 worktree must be clean before an accepted run."
        )
    return branch, commit


def main_goal_commit() -> str:
    return git_value(EXACT_STAGE3_ROOT, ["rev-parse", "refs/remotes/origin/main"])


def require_h100() -> None:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi failed")
    lines = completed.stdout.splitlines()
    first_gpu = lines[0] if lines else ""
    if "H100" not in first_gpu or "9.0" not in first_gpu:
        raise RuntimeError(
            f"Expected an H100 with compute capability 9.0; found {first_gpu!r}."
        )


def run_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_HOME"] = str(CUDA_HOME)
    environment["PATH"] = os.pathsep.join(
        [str(PYTHON_BIN.parent), str(CUDA_HOME / "bin"), environment.get("PATH", "")]
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [
            str(PYTHON_BIN.parents[1] / "lib"),
            str(CUDA_HOME / "lib64"),
            environment.get("LD_LIBRARY_PATH", ""),
        ]
    )
    return environment


def installed_cuml_version(environment: dict[str, str]) -> str:
    completed = subprocess.run(
        [str(PYTHON_BIN), "-c", "import cuml; print(cuml.__version__)"],
        cwd=CUML_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or "Could not read the cuML package version."
        )
    version = completed.stdout.strip()
    if not version.startswith(EXPECTED_CUML_VERSION_PREFIX):
        raise RuntimeError(
            f"Expected cuML {EXPECTED_CUML_VERSION_PREFIX}.x, found {version!r}."
        )
    return version


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_zstd_header(path: Path) -> tuple[int, int]:
    process = subprocess.Popen(
        ["zstd", "-q", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is None:
        raise RuntimeError(f"Could not read {path}")
    first_line = process.stdout.readline()
    process.stdout.close()
    process.terminate()
    process.wait()
    if process.stderr is not None:
        process.stderr.close()
    fields = first_line.split()
    if len(fields) != 2:
        raise ValueError(f"Invalid matrix header in {path}")
    return int(fields[0]), int(fields[1])


def load_ground_truth_manifest() -> dict[str, dict[str, str]]:
    with GROUND_TRUTH_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_workload = {row["workload"]: row for row in rows}
    if (
        len(rows) != len(WORKLOADS)
        or set(by_workload) != {workload.name for workload in WORKLOADS}
    ):
        raise ValueError(
            "The ground-truth manifest workload names do not match the "
            "Stage 3 cuML workload list."
        )
    for workload in WORKLOADS:
        row = by_workload[workload.name]
        expected = {
            "method": "Basic-Scan",
            "precision": PRECISION,
            "scale": SCALE,
            "data_rows": str(workload.data_rows),
            "query_rows": str(workload.query_rows),
            "dimensions": str(workload.dimensions),
            "output_rows": str(workload.query_rows),
            "status": "ok",
        }
        for name, value in expected.items():
            if row.get(name) != value:
                raise ValueError(
                    f"The Work 3 record for {workload.name} has "
                    f"{name}={row.get(name)!r}; expected {value!r}."
                )
    return by_workload


def validate_inputs(
    ground_truth_manifest: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    checksums: dict[str, dict[str, str]] = {}
    for workload in WORKLOADS:
        require_paths(
            [workload.data_path, workload.query_path, workload.reference_path]
        )
        data_shape = read_zstd_header(workload.data_path)
        query_shape = read_zstd_header(workload.query_path)
        expected_data_shape = (workload.data_rows, workload.dimensions)
        expected_query_shape = (workload.query_rows, workload.dimensions)
        if data_shape != expected_data_shape:
            raise ValueError(
                f"{workload.data_path} has shape {data_shape}; "
                f"expected {expected_data_shape}."
            )
        if query_shape != expected_query_shape:
            raise ValueError(
                f"{workload.query_path} has shape {query_shape}; "
                f"expected {expected_query_shape}."
            )

        print(f"[stage3-cuml] hashing {workload.name} inputs", flush=True)
        actual = {
            "data": sha256_file(workload.data_path),
            "query": sha256_file(workload.query_path),
            "reference": sha256_file(workload.reference_path),
        }
        manifest_row = ground_truth_manifest[workload.name]
        expected = {
            "data": manifest_row["data_sha256"],
            "query": manifest_row["query_sha256"],
            "reference": manifest_row["output_zst_sha256"],
        }
        if actual != expected:
            raise ValueError(
                f"{workload.name} input or reference checksum differs "
                "from the accepted Work 3 manifest."
            )
        checksums[workload.name] = actual
    return checksums


def open_vector(path: Path) -> tuple[TextIO, subprocess.Popen[str] | None]:
    if path.name.endswith(".zst"):
        process = subprocess.Popen(
            ["zstd", "-q", "-dc", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if process.stdout is None:
            raise RuntimeError(f"Could not open {path}")
        return process.stdout, process
    return path.open("r", encoding="utf-8"), None


def close_vector(handle: TextIO, process: subprocess.Popen[str] | None) -> None:
    handle.close()
    if process is None:
        return
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            stderr.strip() or f"zstd failed with return code {return_code}"
        )


def iter_floats(path: Path) -> Iterator[float]:
    handle, process = open_vector(path)
    try:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                raise ValueError(f"Blank line {line_number} in {path}")
            for field in fields:
                yield float(field)
    finally:
        close_vector(handle, process)


def compare_values(
    output_values: Iterable[float],
    reference_values: Iterable[float],
    expected_count: int,
) -> dict[str, object]:
    outputs = list(output_values)
    references = list(reference_values)
    output_count = len(outputs)
    reference_count = len(references)
    compared_count = min(output_count, reference_count)
    failure_count = 0
    max_absolute_error = 0.0
    sum_absolute_error = 0.0
    max_relative_error = 0.0
    sum_relative_error = 0.0
    zero_reference_count = 0
    nonzero_output_zero_reference_count = 0

    for output, reference in zip(outputs, references):
        output = float(output)
        reference = float(reference)
        if not math.isfinite(output) or not math.isfinite(reference):
            failure_count += 1
            max_absolute_error = math.inf
            sum_absolute_error = math.inf
            max_relative_error = math.inf
            sum_relative_error = math.inf
            continue

        absolute_error = abs(output - reference)
        max_absolute_error = max(max_absolute_error, absolute_error)
        sum_absolute_error += absolute_error
        if reference == 0.0:
            zero_reference_count += 1
            if output == 0.0:
                relative_error = 0.0
            else:
                relative_error = math.inf
                failure_count += 1
                nonzero_output_zero_reference_count += 1
        else:
            relative_error = absolute_error / abs(reference)
            if relative_error > RELATIVE_TOLERANCE:
                failure_count += 1
        max_relative_error = max(max_relative_error, relative_error)
        sum_relative_error += relative_error

    if (
        output_count != expected_count
        or reference_count != expected_count
        or output_count != reference_count
    ):
        failure_count = max(1, failure_count)
        correctness_status = "count_mismatch"
    else:
        correctness_status = "compared"

    mean_absolute_error = (
        sum_absolute_error / compared_count if compared_count else math.inf
    )
    mean_relative_error = (
        sum_relative_error / compared_count if compared_count else math.inf
    )
    return {
        "correctness": str(
            correctness_status == "compared" and failure_count == 0
        ).lower(),
        "correctness_status": correctness_status,
        "failure_count": failure_count,
        "output_count": output_count,
        "max_abs_err": max_absolute_error,
        "mean_abs_err": mean_absolute_error,
        "max_rel_err": max_relative_error,
        "mean_rel_err": mean_relative_error,
        "zero_reference_count": zero_reference_count,
        "nonzero_output_zero_reference_count": (
            nonzero_output_zero_reference_count
        ),
    }


def compare_output(
    output_path: Path, reference_path: Path, expected_count: int
) -> dict[str, object]:
    require_paths([output_path, reference_path])
    return compare_values(
        iter_floats(output_path), iter_floats(reference_path), expected_count
    )


def parse_metrics(log_text: str) -> dict[str, str]:
    patterns = {
        "cuml_version": r"^cuML package:\s*(.+?)\s*$",
        "timing_mode": r"^Timing mode:\s*(.+?)\s*$",
        "timing_scope": r"^timing_scope:\s*(.+?)\s*$",
        "execution_seconds": r"^execution_seconds:\s*(.+?)\s*$",
        "query_count": r"^query_count:\s*(.+?)\s*$",
        "qps": r"^qps:\s*(.+?)\s*$",
    }
    metrics: dict[str, str] = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, log_text, re.MULTILINE)
        if match:
            metrics[name] = match.group(1)
    return metrics


def validate_metrics(
    metrics: dict[str, str], workload: Workload, expected_version: str
) -> None:
    expected = {
        "cuml_version": expected_version,
        "timing_mode": TIMING_MODE,
        "timing_scope": EXPECTED_TIMING_SCOPE,
        "query_count": str(workload.query_rows),
    }
    for name, value in expected.items():
        if metrics.get(name) != value:
            raise ValueError(
                f"{workload.name} reported {name}={metrics.get(name)!r}; "
                f"expected {value!r}."
            )
    execution_seconds = float(metrics.get("execution_seconds", "nan"))
    reported_qps = float(metrics.get("qps", "nan"))
    if not math.isfinite(execution_seconds) or execution_seconds <= 0.0:
        raise ValueError(
            f"{workload.name} reported invalid execution time: {execution_seconds}"
        )
    if not math.isfinite(reported_qps) or reported_qps <= 0.0:
        raise ValueError(f"{workload.name} reported invalid QPS: {reported_qps}")
    expected_qps = workload.query_rows / execution_seconds
    if not math.isclose(reported_qps, expected_qps, rel_tol=1e-8):
        raise ValueError(
            f"{workload.name} reported QPS {reported_qps}; "
            f"expected {expected_qps} from its execution time."
        )


def make_command(
    workload: Workload, output_path: Path, commit: str
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
        TIMING_MODE,
        "--machine",
        MACHINE,
        "--workload",
        workload.name,
        "--source-commit",
        commit,
    ]


def append_command(command: list[object]) -> None:
    write_header = not COMMANDS_PATH.exists()
    with COMMANDS_PATH.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        handle.write(f"cd {shlex.quote(str(CUML_ROOT))}\n")
        handle.write(
            f"# timeout_seconds: {TIMEOUT_SECONDS}\n{shell_join(command)}\n\n"
        )
    COMMANDS_PATH.chmod(0o755)


def run_command(
    command: list[object], environment: dict[str, str], log_path: Path
) -> tuple[float, str]:
    append_command(command)
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=CUML_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        wall_seconds = time.perf_counter() - start
        partial_output = error.stdout or error.output or ""
        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode("utf-8", errors="replace")
        log_path.write_text(
            f"$ {shell_join(command)}\n"
            f"timeout_seconds: {TIMEOUT_SECONDS}\n"
            f"full_command_wall_seconds: {wall_seconds:.9f}\n\n"
            f"{partial_output}",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Command timed out after {TIMEOUT_SECONDS} seconds; see {log_path}."
        ) from error

    wall_seconds = time.perf_counter() - start
    log_text = (
        f"$ {shell_join(command)}\n"
        f"return_code: {completed.returncode}\n"
        f"full_command_wall_seconds: {wall_seconds:.9f}\n\n"
        f"{completed.stdout or ''}"
    )
    log_path.write_text(log_text, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}; see {log_path}."
        )
    return wall_seconds, log_text


def format_number(value: object) -> str:
    number = float(value)
    return "inf" if math.isinf(number) else f"{number:.17g}"


def write_json_atomic(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".partial")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)


def load_record(
    workload: Workload,
    checksums: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
) -> dict[str, object] | None:
    if not workload.record_path.exists():
        return None
    record = json.loads(workload.record_path.read_text(encoding="utf-8"))
    expected: dict[str, object] = {
        "workload": workload.name,
        "git_branch": branch,
        "git_commit": commit,
        "main_goal_commit": goal_commit,
        "data_sha256": checksums["data"],
        "query_sha256": checksums["query"],
        "reference_sha256": checksums["reference"],
        "status": "ok",
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise RuntimeError(
                f"{workload.record_path} has {name}={record.get(name)!r}; "
                f"expected {value!r}."
            )
    workload.output_path.unlink(missing_ok=True)
    return record


def write_summary(records: list[dict[str, object]]) -> None:
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def write_inventory(
    environment: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
    version: str,
) -> None:
    commands = [
        ["hostname"],
        ["date", "-Is"],
        ["free", "-h"],
        ["df", "-h", str(DATA_ROOT), str(WORKSPACE_ROOT)],
        ["nvidia-smi"],
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv",
        ],
        [str(PYTHON_BIN), "--version"],
        [
            str(PYTHON_BIN),
            "-c",
            (
                "import cupy, cuml, numpy, zstandard; "
                "print('cupy', cupy.__version__); "
                "print('cuml', cuml.__version__); "
                "print('numpy', numpy.__version__); "
                "print('zstandard', zstandard.__version__)"
            ),
        ],
        ["git", "-C", str(CUML_ROOT), "status", "--short", "--branch"],
        ["git", "-C", str(CUML_ROOT), "rev-parse", "HEAD"],
    ]
    with INVENTORY_PATH.open("w", encoding="utf-8") as handle:
        handle.write(f"machine: {MACHINE}\n")
        handle.write(f"hostname: {socket.gethostname()}\n")
        handle.write(f"cuml_root: {CUML_ROOT}\n")
        handle.write(f"python_bin: {PYTHON_BIN}\n")
        handle.write(f"data_root: {DATA_ROOT}\n")
        handle.write(f"ground_truth_root: {GROUND_TRUTH_ROOT}\n")
        handle.write(f"run_root: {RUN_ROOT}\n")
        handle.write(f"git_branch: {branch}\n")
        handle.write(f"git_commit: {commit}\n")
        handle.write(f"main_goal_commit: {goal_commit}\n")
        handle.write(f"cuml_version: {version}\n")
        handle.write(f"relative_tolerance: {RELATIVE_TOLERANCE:.17g}\n")
        handle.write("absolute_tolerance: none\n")
        handle.write("zero_reference_rule: exact_zero\n")
        handle.write(f"timeout_seconds: {TIMEOUT_SECONDS}\n\n")
        for command in commands:
            handle.write(f"$ {shell_join(command)}\n")
            completed = subprocess.run(
                [str(part) for part in command],
                cwd=CUML_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(completed.stdout or "")
            handle.write("\n")


def write_sha256sums() -> None:
    rows = []
    for path in sorted(RUN_ROOT.rglob("*")):
        if path.is_file() and path != CHECKSUM_PATH:
            rows.append(f"{sha256_file(path)}  ./{path.relative_to(RUN_ROOT)}")
    CHECKSUM_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_directories() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def run_workload(
    workload: Workload,
    checksums: dict[str, str],
    branch: str,
    commit: str,
    goal_commit: str,
    version: str,
    environment: dict[str, str],
) -> dict[str, object]:
    workload.output_path.unlink(missing_ok=True)
    command = make_command(workload, workload.output_path, commit)
    print(f"[stage3-cuml] running {workload.name} cold_start", flush=True)
    wall_seconds, log_text = run_command(command, environment, workload.log_path)
    metrics = parse_metrics(log_text)
    validate_metrics(metrics, workload, version)
    correctness = compare_output(
        workload.output_path, workload.reference_path, workload.query_rows
    )
    output_sha256 = sha256_file(workload.output_path)
    record: dict[str, object] = {
        "machine": MACHINE,
        "workload": workload.name,
        "method": METHOD,
        "precision": PRECISION,
        "scale": SCALE,
        "data_rows": workload.data_rows,
        "query_rows": workload.query_rows,
        "dimensions": workload.dimensions,
        "timing_mode": TIMING_MODE,
        "timing_scope": metrics["timing_scope"],
        "cold_execution_seconds": metrics["execution_seconds"],
        "qps": metrics["qps"],
        "correctness": correctness["correctness"],
        "correctness_status": correctness["correctness_status"],
        "failure_count": correctness["failure_count"],
        "output_count": correctness["output_count"],
        "max_abs_err": format_number(correctness["max_abs_err"]),
        "mean_abs_err": format_number(correctness["mean_abs_err"]),
        "max_rel_err": format_number(correctness["max_rel_err"]),
        "mean_rel_err": format_number(correctness["mean_rel_err"]),
        "zero_reference_count": correctness["zero_reference_count"],
        "nonzero_output_zero_reference_count": correctness[
            "nonzero_output_zero_reference_count"
        ],
        "full_command_wall_seconds": f"{wall_seconds:.9f}",
        "cuml_version": version,
        "data_sha256": checksums["data"],
        "query_sha256": checksums["query"],
        "reference_sha256": checksums["reference"],
        "temporary_output_sha256": output_sha256,
        "git_branch": branch,
        "git_commit": commit,
        "main_goal_commit": goal_commit,
        "log_path": str(workload.log_path),
        "command": shell_join(command),
        "accepted_at": datetime.now().astimezone().isoformat(),
        "status": "ok",
    }
    write_json_atomic(workload.record_path, record)
    workload.output_path.unlink()
    return record


def main() -> int:
    if len(sys.argv) != 1:
        raise ValueError(
            "This runner uses hard-coded Stage 3 paths and accepts no arguments."
        )
    require_paths(
        [
            CUML_ROOT,
            EXACT_STAGE3_ROOT,
            HELPER,
            PYTHON_BIN,
            CUDA_HOME,
            DATA_ROOT,
            GROUND_TRUTH_ROOT,
            GROUND_TRUTH_MANIFEST,
            Path("/usr/bin/zstd"),
        ]
    )
    branch, commit = require_clean_stage3()
    goal_commit = main_goal_commit()
    require_h100()
    prepare_directories()
    environment = run_env()
    version = installed_cuml_version(environment)
    write_inventory(environment, branch, commit, goal_commit, version)
    checksums = validate_inputs(load_ground_truth_manifest())

    records: list[dict[str, object]] = []
    for workload in WORKLOADS:
        existing = load_record(
            workload,
            checksums[workload.name],
            branch,
            commit,
            goal_commit,
        )
        if existing is not None:
            print(f"[stage3-cuml] accepted; skipping {workload.name}", flush=True)
            records.append(existing)
        else:
            records.append(
                run_workload(
                    workload,
                    checksums[workload.name],
                    branch,
                    commit,
                    goal_commit,
                    version,
                    environment,
                )
            )
            print(f"[stage3-cuml] accepted {workload.name}", flush=True)
        write_summary(records)
        write_sha256sums()

    OUTPUT_ROOT.rmdir()
    write_sha256sums()
    print(f"[stage3-cuml] run_root={RUN_ROOT}")
    print(f"[stage3-cuml] summary={SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[stage3-cuml][ERROR] {error}", file=sys.stderr)
        raise SystemExit(1)
