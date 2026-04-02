#!/usr/bin/env python3
"""Run the steady-state search benchmark against Mixpeek MVS (production API).

This measures end-to-end search latency through the Mixpeek API, which includes
API routing, embedding generation, LIRE partition scanning, and result ranking.

Usage:
    python -m benchmark.scripts.bench_mvs \
        --api-key mxp_sk_... \
        --queries 100

    # Use an existing namespace (skip setup):
    python -m benchmark.scripts.bench_mvs \
        --api-key mxp_sk_... \
        --namespace-id ns_xxxx \
        --retriever-id ret_xxxx \
        --queries 100
"""

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.engines.mvs.adapter import MVSAdapter
from benchmark.runner import BenchmarkResult, LatencyStats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Sample queries for text-based search benchmarking
SAMPLE_QUERIES = [
    "wireless bluetooth headphones noise canceling",
    "machine learning neural network training",
    "sustainable energy solar panel installation",
    "cloud computing infrastructure deployment",
    "natural language processing text analysis",
    "computer vision image recognition system",
    "database optimization query performance",
    "cybersecurity threat detection system",
    "mobile application development framework",
    "distributed systems fault tolerance",
    "recommendation engine collaborative filtering",
    "time series forecasting prediction",
    "graph neural network knowledge representation",
    "reinforcement learning reward optimization",
    "data pipeline streaming processing",
    "microservices architecture service mesh",
    "container orchestration kubernetes cluster",
    "edge computing low latency inference",
    "federated learning privacy preserving",
    "quantum computing algorithm simulation",
]


def run_mvs_search_benchmark(
    adapter: MVSAdapter,
    num_queries: int,
    concurrency: int,
    queries: list[str],
) -> BenchmarkResult:
    """Run search queries against MVS and collect metrics."""
    latencies: list[float] = []
    errors = 0

    def execute_query(idx: int) -> float:
        query_text = queries[idx % len(queries)]
        start = time.perf_counter()
        resp = adapter._client.post(
            f"{adapter._base_url}/v1/retrievers/{adapter._retriever_id}/execute",
            headers=adapter._headers(),
            json={"inputs": {"query": query_text}},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        return latency_ms

    start = time.perf_counter()

    if concurrency == 1:
        for i in range(num_queries):
            try:
                lat = execute_query(i)
                latencies.append(lat)
            except Exception as e:
                errors += 1
                logger.debug(f"Query error: {e}")
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(execute_query, i): i for i in range(num_queries)}
            for future in as_completed(futures):
                try:
                    lat = future.result()
                    latencies.append(lat)
                except Exception as e:
                    errors += 1
                    logger.debug(f"Query error: {e}")

    total_time = time.perf_counter() - start
    qps = (num_queries - errors) / total_time if total_time > 0 else 0

    sorted_lat = sorted(latencies) if latencies else [0]

    def pct(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    return BenchmarkResult(
        engine="mvs",
        scenario="steady-state",
        num_vectors=0,
        dimensions=768,
        concurrency=concurrency,
        num_queries=num_queries,
        qps=round(qps, 2),
        recall_at_k=0.0,  # Can't compute recall without ground truth for text queries
        latency=LatencyStats(
            p50_ms=round(pct(sorted_lat, 50), 2),
            p95_ms=round(pct(sorted_lat, 95), 2),
            p99_ms=round(pct(sorted_lat, 99), 2),
            p999_ms=round(pct(sorted_lat, 99.9), 2),
            mean_ms=round(statistics.mean(sorted_lat), 2),
            min_ms=round(sorted_lat[0], 2),
            max_ms=round(sorted_lat[-1], 2),
        ),
        total_time_s=round(total_time, 2),
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Mixpeek API key")
    parser.add_argument("--base-url", default="https://api.mixpeek.com")
    parser.add_argument("--namespace-id", help="Existing namespace ID (skip setup)")
    parser.add_argument("--retriever-id", help="Existing retriever ID (skip setup)")
    parser.add_argument("--collection-id", help="Existing collection ID")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="results/")
    args = parser.parse_args()

    adapter = MVSAdapter(api_key=args.api_key, base_url=args.base_url)

    if args.namespace_id and args.retriever_id:
        adapter._namespace_id = args.namespace_id
        adapter._retriever_id = args.retriever_id
        adapter._collection_id = args.collection_id
        logger.info(f"Using existing namespace={args.namespace_id}, retriever={args.retriever_id}")
    else:
        logger.error("--namespace-id and --retriever-id required for MVS benchmark")
        logger.error("Create a namespace with data first, then pass the IDs here")
        sys.exit(1)

    # Warm up
    logger.info(f"Warming up with {args.warmup} queries...")
    for i in range(args.warmup):
        try:
            adapter._client.post(
                f"{adapter._base_url}/v1/retrievers/{adapter._retriever_id}/execute",
                headers=adapter._headers(),
                json={"inputs": {"query": SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]}},
            )
        except Exception as e:
            logger.warning(f"Warmup query failed: {e}")

    # Run at each concurrency level
    all_results = []
    for conc in args.concurrency:
        logger.info(f"\n--- Concurrency: {conc} ---")
        run_results = []
        for run_idx in range(args.runs):
            logger.info(f"  Run {run_idx + 1}/{args.runs}...")
            result = run_mvs_search_benchmark(
                adapter, args.queries, conc, SAMPLE_QUERIES
            )
            run_results.append(result)
            logger.info(
                f"    QPS: {result.qps:,.1f} | "
                f"P50: {result.latency.p50_ms:.1f}ms | "
                f"P99: {result.latency.p99_ms:.1f}ms | "
                f"Errors: {result.errors}"
            )

        # Median by QPS
        run_results.sort(key=lambda r: r.qps)
        median = run_results[len(run_results) // 2]
        all_results.append(median)

    # Print table
    print("\n" + "=" * 90)
    print(f"{'Engine':<8} {'Conc':>6} {'QPS':>10} {'P50ms':>8} {'P95ms':>8} {'P99ms':>8} {'Errors':>8}")
    print("-" * 90)
    for r in all_results:
        print(f"{r.engine:<8} {r.concurrency:>6} {r.qps:>10,.1f} "
              f"{r.latency.p50_ms:>8.1f} {r.latency.p95_ms:>8.1f} {r.latency.p99_ms:>8.1f} "
              f"{r.errors:>8}")
    print("=" * 90)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "steady-state-mvs.json"
    with open(result_file, "w") as f:
        json.dump({"engine": "mvs", "results": [asdict(r) for r in all_results]}, f, indent=2)
    logger.info(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
