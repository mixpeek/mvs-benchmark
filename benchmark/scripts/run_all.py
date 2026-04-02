#!/usr/bin/env python3
"""
Run all benchmark scenarios against all engines.

Usage:
    python benchmark/scripts/run_all.py --scale 10m
    python benchmark/scripts/run_all.py --scale 1b
"""

import argparse
import subprocess
import sys
from datetime import datetime

SCENARIOS = [
    "steady-state",
    "streaming-ingest",
    "memory-constrained",
    "tco",
    "hybrid-search",
    "filtered-search",
    "cold-start",
]

ENGINES = ["mvs", "qdrant", "milvus", "weaviate", "pgvector", "turbopuffer"]

SCENARIO_ARGS = {
    "streaming-ingest": ["--duration", "24h"],
    "memory-constrained": ["--memory-limit", "32g"],
    "tco": ["--target-qps", "100", "--duration", "24h"],
}

SCENARIO_DATASETS = {
    "hybrid-search": "beir",
    "filtered-search": "yfcc-10m",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run all MVS benchmarks")
    parser.add_argument(
        "--scale",
        choices=["10m", "1b"],
        default="10m",
        help="Dataset scale: 10m for quick iteration, 1b for full benchmark",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=ENGINES,
        help="Engines to test (default: all)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=SCENARIOS,
        help="Scenarios to run (default: all)",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = f"cohere-{'10m' if args.scale == '10m' else '1b'}-768"
    timestamp = datetime.utcnow().strftime("%Y-%m")
    output_dir = f"{args.output}/{timestamp}"

    total = len(args.scenarios) * len(args.engines)
    completed = 0

    for scenario in args.scenarios:
        for engine in args.engines:
            completed += 1
            print(f"\n{'='*60}")
            print(f"[{completed}/{total}] {scenario} x {engine}")
            print(f"{'='*60}\n")

            ds = SCENARIO_DATASETS.get(scenario, dataset)
            extra_args = SCENARIO_ARGS.get(scenario, [])

            cmd = [
                sys.executable,
                "benchmark/scripts/run.py",
                "--scenario", scenario,
                "--engine", engine,
                "--dataset", ds,
                "--output", output_dir,
                *extra_args,
            ]

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAILED: {scenario} x {engine}")
                # Continue with other benchmarks

    print(f"\nAll benchmarks complete. Results in {output_dir}/")


if __name__ == "__main__":
    main()
