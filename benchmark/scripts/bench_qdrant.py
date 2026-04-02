#!/usr/bin/env python3
"""Run the steady-state benchmark against a local Qdrant instance.

Usage:
    python -m benchmark.scripts.bench_qdrant --vectors 10000 --queries 1000
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.engines.qdrant.adapter import QdrantAdapter
from benchmark.runner import BenchmarkConfig, run_steady_state, print_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    adapter = QdrantAdapter(url=args.url, api_key=args.api_key)

    config = BenchmarkConfig(
        num_vectors=args.vectors,
        dimensions=args.dimensions,
        num_queries=args.queries,
        top_k=args.top_k,
        concurrency_levels=[1, 10, 32],
        num_runs=args.runs,
    )

    try:
        results = run_steady_state(adapter, config, output_dir=Path(args.output))
        print_results_table(results)
    finally:
        adapter.teardown()


if __name__ == "__main__":
    main()
