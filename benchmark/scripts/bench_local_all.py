#!/usr/bin/env python3
"""Run all feasible benchmarks locally on Mac Studio.

Tests:
1. Qdrant (local Docker) - steady state at 10K, 50K, 100K vectors
2. MVS (Mixpeek prod API) - steady state search latency
3. Qdrant cold-start recovery

Outputs results as JSON and prints summary table.

Usage:
    python -m benchmark.scripts.bench_local_all \
        --mvs-api-key mxp_sk_... \
        --mvs-namespace ns_xxx \
        --mvs-retriever ret_xxx
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.engines.qdrant.adapter import QdrantAdapter
from benchmark.runner import (
    BenchmarkConfig,
    BenchmarkResult,
    LatencyStats,
    generate_dataset,
    compute_ground_truth,
    run_steady_state,
    print_results_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HARDWARE = {
    "machine": "Mac Studio",
    "chip": "Apple M4 Ultra",
    "cores": "28 (20P + 8E)",
    "memory_gb": 96,
    "os": "macOS Sequoia",
    "qdrant": "Docker (local, default config)",
    "mvs": "Mixpeek prod API (https://api.mixpeek.com)",
}

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
    "time series forecasting prediction model",
    "graph neural network knowledge representation",
    "reinforcement learning reward optimization",
    "data pipeline streaming real-time processing",
    "microservices architecture service mesh deployment",
    "container orchestration kubernetes cluster management",
    "edge computing low latency inference serving",
    "federated learning privacy preserving machine learning",
    "quantum computing algorithm simulation research",
    "autonomous vehicle perception planning control",
    "protein structure prediction drug discovery",
    "climate modeling weather prediction simulation",
    "supply chain optimization logistics planning",
    "fraud detection anomaly identification system",
    "sentiment analysis opinion mining customer feedback",
    "speech recognition transcription voice assistant",
    "robotic process automation workflow efficiency",
    "digital twin simulation industrial monitoring",
    "blockchain distributed ledger smart contracts",
]


def run_qdrant_steady_state(output_dir: Path) -> list[BenchmarkResult]:
    """Run Qdrant steady-state at multiple dataset sizes."""
    all_results = []

    for num_vectors in [10_000, 50_000, 100_000]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Qdrant Steady-State: {num_vectors:,} vectors")
        logger.info(f"{'='*60}")

        adapter = QdrantAdapter(url="http://localhost:6333")
        config = BenchmarkConfig(
            num_vectors=num_vectors,
            dimensions=768,
            num_queries=1_000,
            top_k=10,
            concurrency_levels=[1, 10, 32, 64],
            batch_size=500,
            warmup_queries=200,
            num_runs=3,
        )

        try:
            results = run_steady_state(adapter, config, output_dir=output_dir)
            all_results.extend(results)
        finally:
            adapter.teardown()

    return all_results


def run_mvs_search(
    api_key: str,
    namespace_id: str,
    retriever_id: str,
    output_dir: Path,
) -> list[BenchmarkResult]:
    """Run MVS search benchmark via Mixpeek prod API."""
    import httpx

    base_url = "https://api.mixpeek.com"
    client = httpx.Client(timeout=60.0)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-Namespace": namespace_id,
        "Content-Type": "application/json",
    }

    all_results = []

    # Warm up
    logger.info("MVS: Warming up (10 queries)...")
    for i in range(10):
        try:
            client.post(
                f"{base_url}/v1/retrievers/{retriever_id}/execute",
                headers=headers,
                json={"inputs": {"query": SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]}},
            )
        except Exception:
            pass

    for concurrency in [1, 5, 10]:
        logger.info(f"\n--- MVS Concurrency: {concurrency} ---")
        run_results = []

        for run_idx in range(3):
            logger.info(f"  Run {run_idx + 1}/3...")
            latencies = []
            errors = 0
            num_queries = 50

            def execute(idx):
                query = SAMPLE_QUERIES[idx % len(SAMPLE_QUERIES)]
                start = time.perf_counter()
                resp = client.post(
                    f"{base_url}/v1/retrievers/{retriever_id}/execute",
                    headers=headers,
                    json={"inputs": {"query": query}},
                )
                lat = (time.perf_counter() - start) * 1000
                resp.raise_for_status()
                return lat

            start = time.perf_counter()
            if concurrency == 1:
                for i in range(num_queries):
                    try:
                        latencies.append(execute(i))
                    except Exception:
                        errors += 1
            else:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = {pool.submit(execute, i): i for i in range(num_queries)}
                    for f in as_completed(futures):
                        try:
                            latencies.append(f.result())
                        except Exception:
                            errors += 1

            total = time.perf_counter() - start
            qps = (num_queries - errors) / total if total > 0 else 0
            sl = sorted(latencies) if latencies else [0]

            def pct(d, p):
                return d[min(int(len(d) * p / 100), len(d) - 1)]

            result = BenchmarkResult(
                engine="mvs",
                scenario="steady-state",
                num_vectors=0,  # production dataset
                dimensions=768,
                concurrency=concurrency,
                num_queries=num_queries,
                qps=round(qps, 2),
                recall_at_k=0.0,
                latency=LatencyStats(
                    p50_ms=round(pct(sl, 50), 2),
                    p95_ms=round(pct(sl, 95), 2),
                    p99_ms=round(pct(sl, 99), 2),
                    p999_ms=round(pct(sl, 99.9), 2),
                    mean_ms=round(statistics.mean(sl), 2),
                    min_ms=round(sl[0], 2),
                    max_ms=round(sl[-1], 2),
                ),
                total_time_s=round(total, 2),
                errors=errors,
            )
            run_results.append(result)
            logger.info(
                f"    QPS: {result.qps:.1f} | P50: {result.latency.p50_ms:.1f}ms | "
                f"P99: {result.latency.p99_ms:.1f}ms | Errors: {errors}"
            )

        run_results.sort(key=lambda r: r.qps)
        all_results.append(run_results[len(run_results) // 2])

    client.close()

    # Save
    result_file = output_dir / "steady-state-mvs.json"
    with open(result_file, "w") as f:
        json.dump({"engine": "mvs", "results": [asdict(r) for r in all_results]}, f, indent=2)

    return all_results


def run_qdrant_cold_start(output_dir: Path) -> dict:
    """Measure Qdrant cold-start: time from container restart to first successful query."""
    import numpy as np

    logger.info("\n" + "=" * 60)
    logger.info("Qdrant Cold-Start Recovery")
    logger.info("=" * 60)

    # First, insert data so there's something to recover
    adapter = QdrantAdapter(url="http://localhost:6333")
    collection = "bench-cold-start"
    dims = 768
    num_vectors = 50_000

    logger.info(f"Loading {num_vectors:,} vectors for cold-start test...")
    adapter.setup(collection, dims)

    rng = np.random.RandomState(99)
    for batch_start in range(0, num_vectors, 500):
        batch_end = min(batch_start + 500, num_vectors)
        ids = [str(i) for i in range(batch_start, batch_end)]
        vecs = rng.randn(batch_end - batch_start, dims).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / norms
        adapter.insert_batch(ids, vecs)

    adapter.wait_for_index()
    count = adapter.count()
    logger.info(f"Loaded {count:,} vectors. Index ready.")

    # Baseline search latency
    query = rng.randn(dims).astype(np.float32)
    query = query / np.linalg.norm(query)
    baseline_latencies = []
    for _ in range(100):
        m = adapter.search(query, top_k=10)
        baseline_latencies.append(m.latency_ms)
    baseline_p50 = sorted(baseline_latencies)[50]
    logger.info(f"Baseline P50: {baseline_p50:.1f}ms")

    # Restart Qdrant container
    logger.info("Restarting Qdrant container...")
    restart_start = time.perf_counter()
    subprocess.run(["docker", "restart", "server-qdrant-1"], capture_output=True, check=True)

    # Poll until first successful query
    first_query_time = None
    stable_time = None
    recovery_latencies = []

    for attempt in range(300):  # 5 min max
        time.sleep(0.5)
        try:
            from qdrant_client import QdrantClient
            test_client = QdrantClient(url="http://localhost:6333", timeout=2)
            info = test_client.get_collection(collection)
            if info.status.value == "green":
                # Try a search
                results = test_client.query_points(
                    collection_name=collection,
                    query=query.tolist(),
                    limit=10,
                )
                elapsed = time.perf_counter() - restart_start
                if first_query_time is None:
                    first_query_time = elapsed
                    logger.info(f"First query succeeded at {elapsed:.1f}s")

                # Measure latency
                start = time.perf_counter()
                test_client.query_points(
                    collection_name=collection,
                    query=query.tolist(),
                    limit=10,
                )
                lat = (time.perf_counter() - start) * 1000
                recovery_latencies.append(lat)

                # Check if stable (P50 within 2x baseline)
                if len(recovery_latencies) >= 10:
                    recent = sorted(recovery_latencies[-10:])
                    recent_p50 = recent[5]
                    if recent_p50 < baseline_p50 * 2 and stable_time is None:
                        stable_time = time.perf_counter() - restart_start
                        logger.info(f"Stable at {stable_time:.1f}s (P50={recent_p50:.1f}ms vs baseline={baseline_p50:.1f}ms)")
                        break
            test_client.close()
        except Exception:
            pass

    result = {
        "engine": "qdrant",
        "scenario": "cold-start",
        "num_vectors": num_vectors,
        "dimensions": dims,
        "baseline_p50_ms": round(baseline_p50, 2),
        "time_to_first_query_s": round(first_query_time, 2) if first_query_time else None,
        "time_to_stable_s": round(stable_time, 2) if stable_time else None,
        "hardware": HARDWARE,
    }

    result_file = output_dir / "cold-start-qdrant.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Cold-start results saved to {result_file}")

    # Clean up
    try:
        adapter.teardown()
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvs-api-key", required=True)
    parser.add_argument("--mvs-namespace", required=True)
    parser.add_argument("--mvs-retriever", required=True)
    parser.add_argument("--output", default="results/local/")
    parser.add_argument("--skip-qdrant", action="store_true")
    parser.add_argument("--skip-mvs", action="store_true")
    parser.add_argument("--skip-cold-start", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # 1. Qdrant steady-state at multiple scales
    if not args.skip_qdrant:
        qdrant_results = run_qdrant_steady_state(output_dir)
        all_results.extend(qdrant_results)

    # 2. MVS via Mixpeek API
    if not args.skip_mvs:
        mvs_results = run_mvs_search(
            args.mvs_api_key,
            args.mvs_namespace,
            args.mvs_retriever,
            output_dir,
        )
        all_results.extend(mvs_results)

    # 3. Cold-start recovery
    if not args.skip_cold_start:
        cold_start = run_qdrant_cold_start(output_dir)
        print(f"\n{'='*60}")
        print("COLD-START RECOVERY")
        print(f"{'='*60}")
        print(f"  Vectors: {cold_start['num_vectors']:,}")
        print(f"  Baseline P50: {cold_start['baseline_p50_ms']:.1f}ms")
        print(f"  Time to first query: {cold_start.get('time_to_first_query_s', 'N/A')}s")
        print(f"  Time to stable: {cold_start.get('time_to_stable_s', 'N/A')}s")

    # Summary table
    if all_results:
        print_results_table(all_results)

    # Save combined results
    combined = {
        "hardware": HARDWARE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in all_results],
    }
    with open(output_dir / "combined-results.json", "w") as f:
        json.dump(combined, f, indent=2)

    logger.info(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
