# MVS Benchmark

Open-source benchmark suite for vector databases at billion-scale, measuring what production workloads actually care about: **cost, memory efficiency, streaming update stability, and hybrid search quality** — not just QPS on a warm cache.

> **Transparency note:** This benchmark is maintained by [Mixpeek](https://mixpeek.com), the creators of MVS. We designed scenarios that highlight MVS's architectural strengths (BYO storage, LIRE partitions, tantivy BM25). We also test scenarios where MVS loses. Every result is reproducible — run it yourself.

## Why another benchmark?

| Existing benchmark | What it measures well | What it misses |
|---|---|---|
| ANN-Benchmarks | Algorithm recall vs QPS | Only 1M scale, no databases, no cost |
| VectorDBBench | Database QPS, latency, filtered search | Vendor-affiliated (Zilliz), no streaming stability |
| Qdrant Benchmark | Honest vendor comparison | Small scale, no cost or memory metrics |

No benchmark measures **cost per query**, **RAM at 1B scale**, or **latency stability during 24-hour ingestion**. Those are the dimensions that determine whether you can actually run a vector database in production.

## Scenarios

| # | Scenario | What it tests | Who wins |
|---|---|---|---|
| 1 | [Steady-state search](scenarios/01-steady-state/) | Recall vs QPS at 1B vectors | Table stakes — everyone must be competitive |
| 2 | [Streaming ingest](scenarios/02-streaming-ingest/) | P99 latency during 24h continuous insert | Partition-based (LIRE) vs graph-based (HNSW) |
| 3 | [Memory-constrained](scenarios/03-memory-constrained/) | 1B vectors in 32 GB RAM | Disk-native architectures |
| 4 | [12-month TCO](scenarios/04-tco/) | $/million queries over 12 months | BYO storage vs managed services |
| 5 | [Hybrid search](scenarios/05-hybrid-search/) | BM25+dense NDCG@10 on BEIR | Native BM25 (tantivy) vs SPLADE |
| 6 | [Filtered search](scenarios/06-filtered-search/) | Recall at 1%–99% filter selectivity | Posting lists vs graph traversal |
| 7 | [Cold-start recovery](scenarios/07-cold-start/) | Time to serve after process kill | Persistent indexes vs RAM-only |

## Systems tested

- **MVS** (Mixpeek Vector Store) — LIRE partitions + PQ on SSD, tantivy BM25, BYO object storage
- **Qdrant** — HNSW, on-disk option, payload indexes
- **Milvus** — IVF/HNSW, GPU support, distributed
- **Weaviate** — HNSW, hybrid search, modules
- **pgvector + pgvectorscale** — Postgres-native, IVFFlat/HNSW
- **turbopuffer** — S3-native, serverless

Each engine runs in Docker with vendor-suggested configurations. PRs with improved configs are welcome.

## Datasets

| Dataset | Vectors | Dimensions | Source | Use |
|---|---|---|---|---|
| Cohere-1B-768 | 1,000,000,000 | 768 | Wikipedia via Cohere embed-v3 | Primary benchmark |
| DEEP1B | 1,000,000,000 | 96 | Web images | Legacy comparison |
| Cohere-10M-768 | 10,000,000 | 768 | Wikipedia subset | Quick iteration |
| BEIR (multiple) | Varies | 768 | IR evaluation | Hybrid search quality |
| YFCC-10M | 10,000,000 | 192 | Flickr + metadata | Filtered search |

## Quick start

```bash
# Clone
git clone https://github.com/mixpeek/mvs-benchmark.git
cd mvs-benchmark

# Run a single scenario against a single engine (10M scale for quick test)
python benchmark/scripts/run.py \
  --scenario steady-state \
  --engine mvs \
  --dataset cohere-10m-768 \
  --output results/

# Run all scenarios against all engines (requires ~64GB RAM, ~24h)
python benchmark/scripts/run_all.py --scale 1b

# Generate the results site
python benchmark/site/generate.py --results results/ --output docs/
```

## Reproducing results

Every published result includes:
- Exact Docker image SHA
- Engine configuration file
- Hardware spec (default: `r7i.16xlarge` — 64 vCPU, 512 GB RAM, gp3 SSD)
- Raw JSON results
- The git commit of the benchmark code

```bash
# Reproduce a specific published result
python benchmark/scripts/reproduce.py --result results/2026-04/steady-state-mvs.json
```

## Contributing

We actively invite competitor teams to submit optimized configurations:

```bash
# Add or improve an engine config
cp benchmark/engines/mvs/config.yaml benchmark/engines/your-engine/config.yaml
# Edit config, test locally, submit PR
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Methodology

See [METHODOLOGY.md](benchmark/METHODOLOGY.md) for full details on:
- Client-server architecture
- Cache-clearing protocol
- Warm-up procedure
- Percentile calculation
- Hardware standardization
- Statistical significance testing

## License

Apache 2.0 — benchmark code, configurations, and results are all open source.
