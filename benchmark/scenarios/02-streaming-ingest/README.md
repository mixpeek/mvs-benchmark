# Scenario 2: 24-Hour Streaming Ingest

**What it tests:** Search latency stability during continuous data ingestion.

Most benchmarks use "load then query" — unrealistic for production systems that receive continuous writes. This scenario measures whether search SLAs hold during sustained ingestion.

## Setup

- **Initial dataset:** 900M vectors (Cohere-768)
- **Ingest rate:** ~100M vectors over 24 hours (~1,157 inserts/sec)
- **Concurrent search:** 10 search clients running continuously
- **Duration:** 24 hours

## Metrics

Sampled every 60 seconds:

| Metric | How measured |
|---|---|
| P99 search latency | 99th percentile over the last 60s window |
| Recall@10 | Spot-checked every 15 minutes against updated ground truth |
| Insert throughput | Vectors ingested per second |
| Peak RAM | Max RSS |

## Expected outcome

HNSW-based systems will show **latency spikes of 3-4x** during graph maintenance operations (segment merges, HNSW layer updates). LIRE's partition-based approach maintains stable P99.9 because rebalancing operates on small partitions without global locks.

The visualization is a time-series chart: P99 latency on Y-axis, hours on X-axis. Spikes are immediately visible.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario streaming-ingest \
  --engine all \
  --dataset cohere-1b-768 \
  --duration 24h
```
