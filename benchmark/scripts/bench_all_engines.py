#!/usr/bin/env python3
"""Run steady-state benchmark against all available vector databases.

Usage:
    python -m benchmark.scripts.bench_all_engines --vectors 10000 --queries 1000
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.runner import BenchmarkConfig, run_steady_state, print_results_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HARDWARE = {
    "machine": "Mac Studio",
    "chip": "Apple M4 Ultra",
    "cores": "28 (20P + 8E)",
    "memory_gb": 96,
    "os": "macOS Sequoia",
}


def get_adapters() -> list:
    """Import and instantiate all available adapters."""
    adapters = []

    # Qdrant
    try:
        from benchmark.engines.qdrant.adapter import QdrantAdapter
        a = QdrantAdapter(url="http://localhost:6333")
        if a.health_check():
            adapters.append(a)
            logger.info("Qdrant: available")
        else:
            logger.warning("Qdrant: not reachable")
    except Exception as e:
        logger.warning(f"Qdrant: {e}")

    # Weaviate
    try:
        from benchmark.engines.weaviate.adapter import WeaviateAdapter
        a = WeaviateAdapter(url="http://localhost:8079")
        adapters.append(a)
        logger.info("Weaviate: available")
    except Exception as e:
        logger.warning(f"Weaviate: {e}")

    # Chroma
    try:
        from benchmark.engines.chroma.adapter import ChromaAdapter
        a = ChromaAdapter(host="localhost", port=8100)
        adapters.append(a)
        logger.info("Chroma: available")
    except Exception as e:
        logger.warning(f"Chroma: {e}")

    # Milvus
    try:
        from benchmark.engines.milvus.adapter import MilvusAdapter
        a = MilvusAdapter(host="localhost", port=19530)
        adapters.append(a)
        logger.info("Milvus: available")
    except Exception as e:
        logger.warning(f"Milvus: {e}")

    # pgvector
    try:
        from benchmark.engines.pgvector.adapter import PgvectorAdapter
        a = PgvectorAdapter(host="localhost", port=5433)
        adapters.append(a)
        logger.info("pgvector: available")
    except Exception as e:
        logger.warning(f"pgvector: {e}")

    # LanceDB (embedded, always available)
    try:
        from benchmark.engines.lancedb.adapter import LanceDBAdapter
        a = LanceDBAdapter()
        adapters.append(a)
        logger.info("LanceDB: available (embedded)")
    except Exception as e:
        logger.warning(f"LanceDB: {e}")

    return adapters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=50_000)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/all-engines/")
    parser.add_argument("--engines", nargs="+", default=None, help="Only run these engines")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = get_adapters()
    if args.engines:
        adapters = [a for a in adapters if a.name in args.engines]

    if not adapters:
        logger.error("No engines available!")
        sys.exit(1)

    logger.info(f"\nEngines to benchmark: {[a.name for a in adapters]}")
    logger.info(f"Config: {args.vectors:,} vectors, {args.dimensions}d, {args.queries} queries\n")

    config = BenchmarkConfig(
        num_vectors=args.vectors,
        dimensions=args.dimensions,
        num_queries=args.queries,
        top_k=args.top_k,
        concurrency_levels=[1, 10, 32],
        batch_size=500,
        warmup_queries=200,
        num_runs=args.runs,
    )

    all_results = []
    engine_times = {}

    for adapter in adapters:
        logger.info(f"\n{'='*60}")
        logger.info(f"BENCHMARKING: {adapter.name.upper()}")
        logger.info(f"{'='*60}")

        engine_start = time.perf_counter()
        try:
            results = run_steady_state(adapter, config, output_dir=output_dir)
            all_results.extend(results)
            engine_times[adapter.name] = time.perf_counter() - engine_start
        except Exception as e:
            logger.error(f"FAILED: {adapter.name} — {e}")
            engine_times[adapter.name] = -1
        finally:
            try:
                adapter.teardown()
            except Exception:
                pass

    # Print combined results
    if all_results:
        print("\n\n")
        print("=" * 110)
        print(f"  COMBINED RESULTS — {args.vectors:,} vectors, {args.dimensions}d, top_k={args.top_k}")
        print(f"  Hardware: Mac Studio M4 Ultra, 28 cores, 96GB RAM")
        print("=" * 110)
        print_results_table(all_results)

    # Save combined
    combined = {
        "hardware": HARDWARE,
        "config": asdict(config) if hasattr(config, '__dataclass_fields__') else {
            "num_vectors": args.vectors,
            "dimensions": args.dimensions,
            "num_queries": args.queries,
            "top_k": args.top_k,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine_times_s": engine_times,
        "results": [asdict(r) for r in all_results],
    }
    with open(output_dir / "combined-all-engines.json", "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\nAll results saved to {output_dir}/")
    logger.info("\nEngine run times:")
    for eng, t in engine_times.items():
        if t > 0:
            logger.info(f"  {eng}: {t:.1f}s")
        else:
            logger.info(f"  {eng}: FAILED")


if __name__ == "__main__":
    main()
