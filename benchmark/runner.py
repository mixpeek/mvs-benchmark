"""Core benchmark runner — measures QPS, latency percentiles, and recall."""

import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from benchmark.engines.base import EngineAdapter, SearchMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    num_vectors: int = 10_000
    dimensions: int = 768
    num_queries: int = 1_000
    top_k: int = 10
    concurrency_levels: list[int] = field(default_factory=lambda: [1, 10, 32])
    batch_size: int = 100
    warmup_queries: int = 100
    num_runs: int = 3


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    p999_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float


@dataclass
class BenchmarkResult:
    engine: str
    scenario: str
    num_vectors: int
    dimensions: int
    concurrency: int
    num_queries: int
    qps: float
    recall_at_k: float
    latency: LatencyStats
    total_time_s: float
    errors: int


def generate_dataset(num_vectors: int, dimensions: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate random normalized vectors for benchmarking.

    Returns (dataset_vectors, query_vectors) where query vectors are
    the last 10% of the dataset (held out).
    """
    rng = np.random.RandomState(seed)
    num_queries = min(1000, max(100, num_vectors // 10))
    total = num_vectors + num_queries

    vectors = rng.randn(total, dimensions).astype(np.float32)
    # L2-normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    return vectors[:num_vectors], vectors[num_vectors:]


def compute_ground_truth(dataset: np.ndarray, queries: np.ndarray, top_k: int = 10) -> list[list[int]]:
    """Compute exact k-NN via brute force (cosine similarity on normalized vectors = dot product)."""
    logger.info(f"Computing ground truth for {len(queries)} queries against {len(dataset)} vectors...")
    ground_truth = []
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        # Cosine similarity = dot product for normalized vectors
        scores = batch @ dataset.T  # (batch_size, num_vectors)
        # Get top-k indices
        for row in scores:
            top_indices = np.argpartition(row, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            ground_truth.append(top_indices.tolist())
    logger.info("Ground truth computed.")
    return ground_truth


def compute_recall(results: list[str], ground_truth: list[int], top_k: int = 10) -> float:
    """Compute recall@k: fraction of true top-k neighbors found in results."""
    if not ground_truth:
        return 0.0
    gt_set = set(str(g) for g in ground_truth[:top_k])
    result_set = set(results[:top_k])
    return len(gt_set & result_set) / len(gt_set)


def run_queries_concurrent(
    adapter: EngineAdapter,
    query_vectors: np.ndarray,
    concurrency: int,
    num_queries: int,
    ground_truth: list[list[int]] | None = None,
    top_k: int = 10,
) -> BenchmarkResult:
    """Run queries at a given concurrency level and collect metrics."""
    latencies: list[float] = []
    recalls: list[float] = []
    errors = 0

    # Cycle through query vectors if num_queries > len(query_vectors)
    def get_query_idx(i: int) -> int:
        return i % len(query_vectors)

    def execute_query(query_idx: int) -> tuple[float, float]:
        """Execute a single query, return (latency_ms, recall)."""
        vec = query_vectors[query_idx]
        metrics = adapter.search(vec, top_k=top_k)
        recall = 0.0
        if ground_truth and query_idx < len(ground_truth):
            result_ids = [r.id for r in metrics.results]
            recall = compute_recall(result_ids, ground_truth[query_idx], top_k)
        return metrics.latency_ms, recall

    start = time.perf_counter()

    if concurrency == 1:
        # Sequential execution
        for i in range(num_queries):
            try:
                lat, rec = execute_query(get_query_idx(i))
                latencies.append(lat)
                recalls.append(rec)
            except Exception as e:
                errors += 1
                logger.debug(f"Query error: {e}")
    else:
        # Concurrent execution
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(execute_query, get_query_idx(i)): i
                for i in range(num_queries)
            }
            for future in as_completed(futures):
                try:
                    lat, rec = future.result()
                    latencies.append(lat)
                    recalls.append(rec)
                except Exception as e:
                    errors += 1
                    logger.debug(f"Query error: {e}")

    total_time = time.perf_counter() - start
    qps = (num_queries - errors) / total_time if total_time > 0 else 0

    sorted_lat = sorted(latencies) if latencies else [0]

    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    latency_stats = LatencyStats(
        p50_ms=percentile(sorted_lat, 50),
        p95_ms=percentile(sorted_lat, 95),
        p99_ms=percentile(sorted_lat, 99),
        p999_ms=percentile(sorted_lat, 99.9),
        mean_ms=statistics.mean(sorted_lat),
        min_ms=sorted_lat[0],
        max_ms=sorted_lat[-1],
    )

    avg_recall = statistics.mean(recalls) if recalls else 0.0

    return BenchmarkResult(
        engine=adapter.name,
        scenario="steady-state",
        num_vectors=0,  # filled by caller
        dimensions=0,   # filled by caller
        concurrency=concurrency,
        num_queries=num_queries,
        qps=round(qps, 2),
        recall_at_k=round(avg_recall, 4),
        latency=latency_stats,
        total_time_s=round(total_time, 2),
        errors=errors,
    )


def run_steady_state(
    adapter: EngineAdapter,
    config: BenchmarkConfig,
    output_dir: Path | None = None,
) -> list[BenchmarkResult]:
    """Run the steady-state benchmark scenario.

    1. Generate synthetic dataset
    2. Insert vectors
    3. Compute ground truth
    4. Warm up
    5. Run queries at each concurrency level
    6. Report results
    """
    logger.info(f"=== Steady-State Benchmark: {adapter.name} ===")
    logger.info(f"Config: {config.num_vectors} vectors, {config.dimensions}d, "
                f"{config.num_queries} queries, top_k={config.top_k}")

    # Generate dataset
    logger.info("Generating dataset...")
    dataset, queries = generate_dataset(config.num_vectors, config.dimensions)
    logger.info(f"Dataset: {dataset.shape}, Queries: {queries.shape}")

    # Compute ground truth
    ground_truth = compute_ground_truth(dataset, queries, config.top_k)

    # Setup engine
    collection_name = f"bench-steady-state-{config.num_vectors}"
    logger.info(f"Setting up {adapter.name}...")
    adapter.setup(collection_name, config.dimensions)

    # Insert vectors in batches
    logger.info(f"Inserting {config.num_vectors} vectors...")
    insert_start = time.perf_counter()
    for batch_start in range(0, config.num_vectors, config.batch_size):
        batch_end = min(batch_start + config.batch_size, config.num_vectors)
        ids = [str(i) for i in range(batch_start, batch_end)]
        vecs = dataset[batch_start:batch_end]
        adapter.insert_batch(ids, vecs)
        if batch_start % (config.batch_size * 10) == 0:
            logger.info(f"  Inserted {batch_start}/{config.num_vectors}...")
    insert_time = time.perf_counter() - insert_start
    logger.info(f"Insert complete in {insert_time:.1f}s "
                f"({config.num_vectors / insert_time:.0f} vectors/sec)")

    # Wait for indexing
    logger.info("Waiting for index build...")
    adapter.wait_for_index()
    count = adapter.count()
    logger.info(f"Index ready. {count} vectors indexed.")

    # Warm-up
    logger.info(f"Warming up ({config.warmup_queries} queries)...")
    for i in range(config.warmup_queries):
        adapter.search(queries[i % len(queries)], top_k=config.top_k)

    # Run benchmark at each concurrency level
    all_results = []
    for concurrency in config.concurrency_levels:
        logger.info(f"\n--- Concurrency: {concurrency} ---")
        run_results = []
        for run_idx in range(config.num_runs):
            logger.info(f"  Run {run_idx + 1}/{config.num_runs}...")
            result = run_queries_concurrent(
                adapter,
                queries,
                concurrency=concurrency,
                num_queries=config.num_queries,
                ground_truth=ground_truth,
                top_k=config.top_k,
            )
            result.num_vectors = config.num_vectors
            result.dimensions = config.dimensions
            run_results.append(result)

            logger.info(
                f"    QPS: {result.qps:,.1f} | "
                f"Recall@{config.top_k}: {result.recall_at_k:.4f} | "
                f"P50: {result.latency.p50_ms:.1f}ms | "
                f"P99: {result.latency.p99_ms:.1f}ms | "
                f"Errors: {result.errors}"
            )

        # Use median run by QPS
        run_results.sort(key=lambda r: r.qps)
        median_result = run_results[len(run_results) // 2]
        all_results.append(median_result)

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"steady-state-{adapter.name}.json"
        with open(result_file, "w") as f:
            json.dump(
                {
                    "engine": adapter.name,
                    "scenario": "steady-state",
                    "config": {
                        "num_vectors": config.num_vectors,
                        "dimensions": config.dimensions,
                        "num_queries": config.num_queries,
                        "top_k": config.top_k,
                    },
                    "results": [asdict(r) for r in all_results],
                },
                f,
                indent=2,
            )
        logger.info(f"\nResults saved to {result_file}")

    return all_results


def print_results_table(results: list[BenchmarkResult]) -> None:
    """Pretty-print results as a table."""
    print("\n" + "=" * 100)
    print(f"{'Engine':<12} {'Concurrency':>12} {'QPS':>10} {'Recall@10':>10} "
          f"{'P50ms':>8} {'P95ms':>8} {'P99ms':>8} {'Errors':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r.engine:<12} {r.concurrency:>12} {r.qps:>10,.1f} {r.recall_at_k:>10.4f} "
              f"{r.latency.p50_ms:>8.1f} {r.latency.p95_ms:>8.1f} {r.latency.p99_ms:>8.1f} "
              f"{r.errors:>8}")
    print("=" * 100)
