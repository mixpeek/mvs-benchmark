# Scenario 1: Steady-State Search

**What it tests:** Baseline recall vs QPS tradeoff at billion-scale.

This is the table-stakes benchmark that every vector database must be competitive on. It measures raw search performance on a static, fully-indexed dataset.

## Setup

- **Dataset:** Cohere-1B-768 (1B vectors, 768 dimensions)
- **RAM:** Unconstrained (use what you need)
- **Queries:** 10,000 held-out vectors
- **Concurrency:** 1, 10, 32, 64, 100 parallel clients

## Metrics

| Metric | How measured |
|---|---|
| Recall@10 | Fraction of true 10-NN in results |
| QPS | Queries per second at steady state |
| P50 latency | Median query latency |
| P95 latency | 95th percentile query latency |
| P99 latency | 99th percentile query latency |
| Peak RAM | Max RSS during query phase |

## Expected outcome

HNSW-based systems (Qdrant, Milvus, Weaviate) will likely lead on raw QPS for fully RAM-resident workloads. MVS should be competitive but not necessarily leading. The key insight comes when combining this scenario with Scenarios 3 (memory) and 4 (cost).

## Run

```bash
python benchmark/scripts/run.py \
  --scenario steady-state \
  --engine all \
  --dataset cohere-1b-768
```
