# Scenario 7: Cold-Start Recovery

**What it tests:** Time from process start to serving queries after a crash or restart.

In production, processes restart — deployments, OOM kills, node failures. Recovery time determines your availability SLA. Systems that require loading entire indexes into RAM before serving have fundamentally different recovery profiles than disk-native systems.

## Setup

- **Dataset:** Cohere-1B-768 (1B vectors, 768 dimensions), fully indexed
- **Procedure:**
  1. Engine is running and serving queries at steady state
  2. `kill -9` the engine process (simulates OOM kill)
  3. Restart the engine
  4. Measure time until first successful query
  5. Measure time until P99 latency stabilizes within 2x of steady state

## Recovery phases

| Phase | What happens |
|---|---|
| Process start | Engine binary starts, reads config |
| Index load | Load index structures into memory |
| Warm-up | Page faults settle, caches warm |
| Stable | Performance within 2x of pre-crash steady state |

## Metrics

| Metric | How measured |
|---|---|
| Time to first query | Wall clock from process start to first successful response |
| Time to stable P99 | Wall clock until P99 latency is within 2x of steady-state baseline |
| RAM during recovery | Memory usage curve during startup |
| Query success rate | % of queries that succeed during recovery (vs timeout/error) |

## Expected outcome

RAM-resident systems (HNSW) must reload the full graph into memory before serving. At 1B scale with 3+ TB of index data, this takes **30-60 minutes** even on fast SSDs. Disk-native systems (MVS, turbopuffer, DiskANN) can serve queries within **seconds** because they read from disk on-demand.

This scenario makes the availability cost of RAM-resident architectures visible.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario cold-start \
  --engine all \
  --dataset cohere-1b-768
```
