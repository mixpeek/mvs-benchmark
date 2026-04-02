# Scenario 4: 12-Month Total Cost of Ownership

**What it tests:** Real-world cost to operate a vector database at billion-scale for a year.

Most benchmarks report QPS but never ask: "what does this cost?" This scenario computes $/million queries across compute, storage, memory, and operational overhead.

## Setup

- **Dataset:** Cohere-1B-768 (1B vectors, 768 dimensions)
- **Target SLA:** P99 < 100ms, Recall@10 >= 95%
- **Query rate:** 100 QPS sustained (8.64M queries/day)
- **Ingestion:** 1M new vectors/day
- **Retention:** All vectors retained for 12 months

## Cost model

| Component | How measured |
|---|---|
| Compute | Instance cost to sustain target QPS at target recall |
| Memory | RAM required (determines instance family/size) |
| Storage | Disk + object storage for vectors and indexes |
| Network | Cross-AZ/region transfer for replication |
| Operational | Human hours for maintenance, upgrades, scaling events |

### Instance sizing methodology

Each engine is given its **minimum viable instance** — the cheapest instance type that meets the SLA. We do not artificially constrain all engines to the same instance; that would penalize memory-efficient engines.

| System | Expected instance | Monthly compute |
|---|---|---|
| HNSW (full RAM) | r7i.metal-48xl (1.5 TB) | ~$17,000 |
| HNSW + PQ | r7i.8xlarge (256 GB) | ~$2,400 |
| DiskANN | r7i.4xlarge (128 GB) | ~$1,200 |
| MVS (BYO S3) | c7i.4xlarge (32 GB) + S3 | ~$600 |

## Metrics

| Metric | How measured |
|---|---|
| $/million queries | Total monthly cost / queries served |
| Annual TCO | 12-month projection including storage growth |
| Cost at scale | How cost changes from 100M → 1B → 10B vectors |
| Break-even point | Dataset size where disk-native beats RAM-resident |

## Expected outcome

BYO-storage architectures (MVS, turbopuffer) will show 5-10x cost advantage at billion-scale because they avoid paying for RAM to hold vectors. The gap widens as dataset size grows — RAM cost scales linearly, object storage cost is near-flat.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario tco \
  --engine all \
  --dataset cohere-1b-768 \
  --duration 24h \
  --target-qps 100
```
