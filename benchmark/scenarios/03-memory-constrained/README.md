# Scenario 3: Memory-Constrained Billion-Scale

**What it tests:** Search performance when RAM is scarce relative to dataset size.

This is where architectural differences become stark. At 1B x 768-dim, raw vectors alone are ~3 TB. HNSW needs vectors + graph in RAM. Disk-native architectures operate natively under memory pressure.

## Setup

- **Dataset:** Cohere-1B-768 (1B vectors, 768 dimensions)
- **RAM limit:** 32 GB (via cgroup)
- **Queries:** 10,000 held-out vectors, 10 concurrent clients

## Memory math

| System | Estimated RAM need (1B x 768-dim) |
|---|---|
| HNSW (full) | ~3.3 TB (vectors + graph) |
| HNSW + PQ | ~520 GB (compressed vectors + graph) |
| DiskANN | ~32 GB (PQ codes in RAM, graph on SSD) |
| LIRE/MVS | ~10 GB (partition centroids + PQ codebook) |

## Metrics

| Metric | How measured |
|---|---|
| Recall@10 | At various QPS levels |
| QPS | At 95% recall target |
| P99 latency | At 10 concurrent clients |
| OOM events | Whether the engine crashes or degrades gracefully |

## Expected outcome

HNSW-based systems will either fail to load (OOM), fall back to on-disk mode with degraded performance, or require PQ compression with recall loss. Systems designed for disk-first operation (MVS, turbopuffer, DiskANN) will function normally.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario memory-constrained \
  --engine all \
  --dataset cohere-1b-768 \
  --memory-limit 32g
```
