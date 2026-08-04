#!/usr/bin/env python3
"""Run original cuML KDE on deterministic unscaled Stage 4 inputs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from cuml_stage1p1_exact_kde import CUML_BANDWIDTH, import_cuml, run_pipeline
from stage4_synthetic_common import (
    GENERATOR_NAME,
    OUTPUT_HASH_NAME,
    generate_inputs,
    output_hash,
    positive_int32,
    uint64_seed,
    write_output,
)


TIMING_SCOPE = "generated_host_inputs_to_host_output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original cuML on deterministic Stage 4 synthetic input."
    )
    parser.add_argument("data_rows", type=positive_int32)
    parser.add_argument("query_rows", type=positive_int32)
    parser.add_argument("dimension", type=positive_int32)
    parser.add_argument("seed", type=uint64_seed)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import_start = time.perf_counter()
    cp, cuml, KernelDensity = import_cuml()
    import_seconds = time.perf_counter() - import_start

    generation_start = time.perf_counter()
    data, queries = generate_inputs(
        args.data_rows,
        args.query_rows,
        args.dimension,
        args.seed,
    )
    generation_seconds = time.perf_counter() - generation_start

    execution_start = time.perf_counter()
    output = run_pipeline(
        cp,
        KernelDensity,
        data,
        queries,
        None,
        float(args.data_rows),
    )
    execution_seconds = time.perf_counter() - execution_start

    if args.output is not None:
        write_output(args.output, output)
    qps = args.query_rows / execution_seconds

    print("implementation: original-cuml")
    print("precision: fp64")
    print("input_source: synthetic")
    print(f"data_rows: {args.data_rows}")
    print(f"query_count: {args.query_rows}")
    print(f"dimension: {args.dimension}")
    print("weights: unit")
    print("scale_mode: none")
    print("effective_gamma: 1")
    print(f"synthetic_generator: {GENERATOR_NAME}")
    print(f"synthetic_seed: {args.seed}")
    print(f"input_generation_seconds: {generation_seconds:.9f}")
    print(f"library_import_seconds: {import_seconds:.9f}")
    print(f"cuml_version: {cuml.__version__}")
    print(f"kernel_density_bandwidth: {CUML_BANDWIDTH:.17g}")
    print("timing_mode: cold_start")
    print(f"timing_scope: {TIMING_SCOPE}")
    print(f"execution_seconds: {execution_seconds:.9f}")
    print(f"qps: {qps:.9f}")
    print(f"output_count: {output.size}")
    print(f"output_hash_algorithm: {OUTPUT_HASH_NAME}")
    print(f"output_hash: {output_hash(output)}")
    print(f"output_destination: {'file' if args.output is not None else 'none'}")
    if args.output is not None:
        print(f"output_file: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
