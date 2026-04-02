#!/usr/bin/env python3
"""
MVS Benchmark Runner

Runs a single benchmark scenario against one or more engines.

Usage:
    python benchmark/scripts/run.py \
        --scenario steady-state \
        --engine mvs \
        --dataset cohere-10m-768 \
        --output results/
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCENARIOS = {
    "steady-state": "01-steady-state",
    "streaming-ingest": "02-streaming-ingest",
    "memory-constrained": "03-memory-constrained",
    "tco": "04-tco",
    "hybrid-search": "05-hybrid-search",
    "filtered-search": "06-filtered-search",
    "cold-start": "07-cold-start",
}

ENGINES = ["mvs", "qdrant", "milvus", "weaviate", "pgvector", "turbopuffer"]

DATASETS = {
    "cohere-1b-768": {"vectors": 1_000_000_000, "dimensions": 768},
    "cohere-10m-768": {"vectors": 10_000_000, "dimensions": 768},
    "deep1b": {"vectors": 1_000_000_000, "dimensions": 96},
    "beir": {"vectors": None, "dimensions": 768},  # multiple sub-datasets
    "yfcc-10m": {"vectors": 10_000_000, "dimensions": 192},
}


def parse_args():
    parser = argparse.ArgumentParser(description="MVS Benchmark Runner")
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Benchmark scenario to run",
    )
    parser.add_argument(
        "--engine",
        required=True,
        help="Engine to benchmark (or 'all')",
    )
    parser.add_argument(
        "--dataset",
        default="cohere-10m-768",
        choices=list(DATASETS.keys()),
        help="Dataset to use",
    )
    parser.add_argument(
        "--output",
        default="results/",
        help="Output directory for results",
    )
    parser.add_argument(
        "--memory-limit",
        default=None,
        help="Memory limit (e.g., '32g') — used for memory-constrained scenario",
    )
    parser.add_argument(
        "--duration",
        default=None,
        help="Duration (e.g., '24h') — used for streaming-ingest scenario",
    )
    parser.add_argument(
        "--target-qps",
        type=int,
        default=None,
        help="Target QPS — used for TCO scenario",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 10, 32, 64, 100],
        help="Concurrency levels to test",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of independent runs (median reported)",
    )
    return parser.parse_args()


def load_engine_config(engine: str) -> dict:
    """Load engine configuration from YAML."""
    config_path = Path(f"benchmark/engines/{engine}/config.yaml")
    if not config_path.exists():
        logger.error(f"No config found for engine '{engine}' at {config_path}")
        sys.exit(1)

    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        logger.error("PyYAML required: pip install pyyaml")
        sys.exit(1)


def start_engine(engine: str, config: dict, memory_limit: str | None = None) -> str:
    """Start the engine Docker container. Returns container ID."""
    image = config["engine"]["docker_image"]
    name = f"mvs-bench-{engine}"

    cmd = [
        "docker", "run", "-d",
        "--name", name,
        "-v", "/data:/data",
        "--network", "host",
    ]

    if memory_limit:
        cmd.extend(["--memory", memory_limit])

    cmd.append(image)

    logger.info(f"Starting {engine}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to start {engine}: {result.stderr}")
        sys.exit(1)

    return result.stdout.strip()


def wait_for_engine(engine: str, config: dict, timeout: int = 300):
    """Wait for engine to be ready to accept connections."""
    logger.info(f"Waiting for {engine} to be ready...")
    # Engine-specific health check logic would go here
    time.sleep(5)  # placeholder


def load_dataset(engine: str, dataset: str, config: dict):
    """Load dataset into the engine."""
    logger.info(f"Loading {dataset} into {engine}...")
    # Dataset loading logic would go here
    # This would use engine-specific clients to bulk-insert vectors


def run_warmup(engine: str, config: dict, num_queries: int = 1000):
    """Run warm-up queries (results discarded)."""
    logger.info(f"Running {num_queries} warm-up queries on {engine}...")
    # Warm-up logic would go here


def clear_page_cache():
    """Clear OS page cache for fair cold-cache comparison."""
    logger.info("Clearing page cache...")
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.warning("Could not clear page cache (requires root)")


def run_queries(
    engine: str,
    config: dict,
    dataset: str,
    concurrency: int,
    num_queries: int = 10000,
) -> dict:
    """Run benchmark queries and collect metrics."""
    logger.info(
        f"Running {num_queries} queries on {engine} "
        f"with concurrency={concurrency}..."
    )

    # This is where the actual benchmark logic would go:
    # 1. Load query vectors from dataset
    # 2. Load ground truth
    # 3. Run queries with async client at given concurrency
    # 4. Compute recall, latency percentiles, QPS

    # Placeholder return
    return {
        "engine": engine,
        "dataset": dataset,
        "concurrency": concurrency,
        "num_queries": num_queries,
        "metrics": {
            "qps": 0,
            "recall_at_10": 0,
            "p50_ms": 0,
            "p95_ms": 0,
            "p99_ms": 0,
            "p999_ms": 0,
            "peak_ram_gb": 0,
        },
    }


def stop_engine(engine: str):
    """Stop and remove the engine container."""
    name = f"mvs-bench-{engine}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def main():
    args = parse_args()

    engines = ENGINES if args.engine == "all" else [args.engine]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")

    for engine in engines:
        logger.info(f"=== Benchmarking {engine} on {args.scenario} ===")

        config = load_engine_config(engine)

        all_runs = []
        for run_idx in range(args.runs):
            logger.info(f"--- Run {run_idx + 1}/{args.runs} ---")

            # Start engine
            container_id = start_engine(engine, config, args.memory_limit)

            try:
                wait_for_engine(engine, config)
                load_dataset(engine, args.dataset, config)
                run_warmup(engine, config)
                clear_page_cache()
                run_warmup(engine, config)  # second warm-up after cache clear

                # Run at each concurrency level
                run_results = []
                for concurrency in args.concurrency:
                    result = run_queries(
                        engine, config, args.dataset, concurrency
                    )
                    run_results.append(result)

                all_runs.append(run_results)
            finally:
                stop_engine(engine)

        # Save results
        result_file = output_dir / f"{args.scenario}-{engine}-{timestamp}.json"
        result_data = {
            "scenario": args.scenario,
            "engine": engine,
            "dataset": args.dataset,
            "timestamp": timestamp,
            "config": config,
            "runs": all_runs,
            "hardware": {
                "instance": "r7i.16xlarge",
                "vcpus": 64,
                "ram_gb": 512,
                "disk": "gp3 SSD",
            },
        }

        with open(result_file, "w") as f:
            json.dump(result_data, f, indent=2)

        logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
