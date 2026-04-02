# Scenario 6: Filtered Search at Variable Selectivity

**What it tests:** How recall and QPS degrade as metadata filters become more or less selective.

Filtered search is the most common production query pattern — "find similar items WHERE category = X AND price < Y." Filter selectivity (what fraction of the dataset passes the filter) dramatically affects different index architectures.

## Setup

- **Dataset:** YFCC-10M (10M vectors, 192 dimensions, rich metadata)
- **Metadata fields:** category (100 values), country (200 values), year (20 values), has_geo (boolean)
- **Filter selectivities tested:** 0.1%, 1%, 5%, 10%, 25%, 50%, 75%, 99%
- **Queries:** 10,000 per selectivity level

## Filter strategies by architecture

| Strategy | How it works | Good at |
|---|---|---|
| Pre-filter | Filter first, then ANN on subset | High selectivity (small subset) |
| Post-filter | ANN first, then filter results | Low selectivity (most data passes) |
| In-graph filter | Filter during HNSW traversal | Mid-range selectivity |
| Partition-scoped | Scan relevant partitions with filter | Uniform across selectivities |

## Metrics

| Metric | How measured |
|---|---|
| Recall@10 | At each selectivity level |
| QPS | At each selectivity level |
| P99 latency | At each selectivity level |
| Recall cliff | Selectivity where recall drops below 90% |

## Expected outcome

HNSW post-filter struggles at high selectivity (1%) — it must over-fetch massively to find enough matching results. Pre-filter at low selectivity (99%) is wasteful. Partition-based systems (MVS/LIRE) maintain more uniform performance because partitions naturally co-locate similar vectors and scanning a partition with a filter is cheap.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario filtered-search \
  --engine all \
  --dataset yfcc-10m
```
