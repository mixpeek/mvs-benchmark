# Benchmark Methodology

## Hardware

All benchmarks run on identical hardware unless the scenario specifically tests cost (Scenario 4).

**Default instance:** AWS `r7i.16xlarge`
- 64 vCPUs (Intel Xeon Sapphire Rapids)
- 512 GB RAM (artificially constrained per scenario)
- 2 TB gp3 SSD (16,000 IOPS, 1,000 MB/s)
- Ubuntu 22.04, kernel 6.5+

Memory is constrained via cgroups, not by using a smaller instance. This ensures CPU, disk, and network are identical across all memory configurations.

## Client-server architecture

```
┌─────────────┐         ┌──────────────────┐
│  Benchmark  │  gRPC/  │   Engine Under   │
│   Client    │──HTTP──▶│      Test        │
│  (Python)   │         │   (Docker)       │
└─────────────┘         └──────────────────┘
     │                         │
     │  metrics.json           │  data volume
     ▼                         ▼
  results/               /data (gp3 SSD)
```

- Client and server run on the **same machine** (no network variance)
- Client runs outside the engine's cgroup (doesn't compete for constrained RAM)
- Client uses async HTTP/gRPC with configurable concurrency

## Warm-up protocol

1. Load dataset into engine
2. Wait for all background index builds to complete (engine-specific check)
3. Run 1,000 random queries (results discarded)
4. Clear OS page cache: `echo 3 > /proc/sys/vm/drop_caches`
5. Run 1,000 more warm-up queries (results discarded)
6. Begin measurement

For **cold-start scenarios** (Scenario 7), skip steps 3-5.

## Measurement

- **QPS:** Total queries completed / wall-clock seconds, measured at steady state (after 10s ramp-up)
- **Latency:** Per-query wall-clock time, reported as P50, P95, P99, P99.9
- **Recall@K:** Fraction of true K-nearest neighbors returned, computed against brute-force ground truth
- **Memory:** Peak RSS of engine process, sampled every 1s via `/proc/{pid}/status`
- **Disk:** Total bytes in engine data directory after index build

Each measurement run executes **10,000 queries** (or 60 seconds, whichever is longer). Results are the median of **3 independent runs** with fresh data loads.

## Query set

- 10,000 query vectors held out from the dataset (never indexed)
- Queries are issued in random order with uniform distribution
- Concurrency levels: 1, 10, 32, 64, 100 parallel clients

## Ground truth

Brute-force exact K-NN computed offline using NumPy/FAISS exact search. Ground truth files are published alongside datasets for independent verification.

## Engine configuration

Each engine ships with a **default** config (out-of-box settings) and an **optimized** config (tuned for the benchmark). Both are tested and reported. Optimized configs are welcome as PRs from engine teams.

Configs are YAML files in `benchmark/engines/{engine}/config.yaml` with comments explaining each parameter choice.

## Versioning

- Engine versions are pinned to specific Docker image SHAs
- Benchmarks are re-run monthly against latest stable releases
- Historical results are preserved (never overwritten)
- Version bumps are tracked in `CHANGELOG.md`

## Statistical rigor

- Error bars show min/max across 3 runs
- If variance exceeds 10%, run 5 additional iterations
- Report coefficient of variation for all metrics
- No cherry-picking: all runs are published, including bad ones
