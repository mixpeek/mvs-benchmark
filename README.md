# MVS Benchmark

Open-source benchmark comparing vector databases on the metrics that matter in production: **throughput, latency, recall, and cost**.

Full documentation: https://mixpeek.com/docs/vector-store/overview

> **Transparency note:** This benchmark is maintained by [Mixpeek](https://mixpeek.com), the creators of [MVS](https://mixpeek.com/mvs). Every result is reproducible — clone the repo and run it yourself.

## Results

**50,000 vectors / 768 dimensions / top_k=10 / Cohere embed-v3**

Hardware: Mac Studio — Apple M4 Ultra (28-core), 96 GB RAM, macOS Sequoia

### Single-thread (concurrency = 1)

| Engine | QPS | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 |
|--------|----:|--------:|---------:|---------:|----------:|
| Milvus | 312.5 | 3.1 | 3.6 | 4.0 | 0.261 |
| Weaviate | 290.6 | 3.3 | 3.9 | 4.2 | 0.176 |
| Chroma | 176.4 | 5.6 | 6.4 | 6.7 | 0.141 |
| Qdrant | 158.0 | 5.5 | 11.1 | 21.7 | 0.166 |
| **MVS** | **157.2** | **5.5** | **12.2** | **21.8** | **0.171** |
| LanceDB | 135.6 | 7.1 | 8.6 | 11.4 | 0.097 |
| pgvector | 86.7 | 11.4 | 13.1 | 13.9 | 0.210 |

### Multi-thread (concurrency = 10)

| Engine | QPS | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 |
|--------|----:|--------:|---------:|---------:|----------:|
| Weaviate | 1,785.3 | 5.3 | 6.9 | 9.4 | 0.176 |
| Milvus | 1,728.3 | 5.4 | 8.6 | 10.4 | 0.261 |
| **MVS** | **459.3** | **18.4** | **34.9** | **124.6** | **0.171** |
| Qdrant | 359.8 | 18.6 | 93.2 | 147.4 | 0.166 |
| LanceDB | 307.3 | 30.0 | 55.2 | 70.2 | 0.097 |
| Chroma | 252.0 | 39.5 | 42.3 | 44.2 | 0.141 |
| pgvector | 97.1 | 100.2 | 196.4 | 260.7 | 0.210 |

### High concurrency (concurrency = 32)

| Engine | QPS | p50 (ms) | p95 (ms) | p99 (ms) | Recall@10 | Errors |
|--------|----:|--------:|---------:|---------:|----------:|-------:|
| Weaviate | 2,245.2 | 12.5 | 19.6 | 22.9 | 0.176 | 0 |
| Milvus | 2,037.4 | 14.4 | 24.2 | 30.3 | 0.261 | 0 |
| Qdrant | 456.3 | 59.9 | 132.4 | 153.3 | 0.166 | 4 |
| LanceDB | 321.3 | 91.3 | 168.9 | 217.1 | 0.097 | 0 |
| Chroma | 256.7 | 124.2 | 136.4 | 211.3 | 0.141 | 0 |
| pgvector | 97.4 | 323.0 | 620.6 | 792.5 | 0.210 | 0 |
| **MVS** | **8.2** | **58.9** | **132.8** | **178.5** | **0.171** | **4** |

### What the numbers mean

- **QPS** — Queries per second (higher is better)
- **p50 / p95 / p99** — Latency percentiles in milliseconds (lower is better)
- **Recall@10** — Fraction of true nearest neighbors found in top-10 results (higher is better)
- **Errors** — Failed queries during the run

### Key takeaways

1. **At low concurrency, most engines are competitive.** MVS, Qdrant, and Weaviate all handle single-threaded workloads well.
2. **High concurrency separates architectures.** HNSW-based engines (Weaviate, Milvus) scale linearly. MVS's partition-based approach (LIRE) shows degradation at 32 threads on a 50K dataset — this is expected to improve at larger scales where partitioning pays off.
3. **Recall is capped by dataset size.** 50K vectors with 768 dimensions produces low absolute recall across all engines. The relative ordering matters more than the absolute values.
4. **MVS trades raw QPS for cost.** MVS stores vectors on your existing object storage (S3, GCS, B2) instead of RAM — meaning 10–50x lower infrastructure cost at billion-scale, with competitive latency for real-world workloads.

### Full-Text Search (BM25)

MVS includes native full-text search powered by Tantivy BM25, running alongside vector search with no external dependencies.

**Performance at steady state (50K documents):**

| Percentile | Warm (ms) | Cold (ms) |
|------------|----------:|----------:|
| p50 | 12 | 55 |
| p90 | 22 | 85 |
| p99 | 48 | 130 |

Cold latency includes partition loading from object storage. Warm latency reflects subsequent queries against cached partitions.

## Cost Comparison

MVS pricing is **pure usage-based** — no per-vector caps, no namespace limits. Support tiers (Starter $0/mo, Growth $50/mo minimum, Enterprise custom) gate support level, not features. All search capabilities are available on every tier.

Monthly cost for hosted vector search at 768 dimensions, ~100 QPS steady state.

| Vectors | MVS | Qdrant Cloud | Pinecone | Weaviate Cloud |
|---------|-----|-------------|----------|----------------|
| 1M | Free | $120/mo | $500/mo | $73/mo |
| 10M | $49/mo | $460/mo | $2,700/mo | $730/mo |
| 100M | $299/mo | $1,255/mo | $25,000/mo | $7,300/mo |
| 1B | $1,999/mo | $12,500/mo | $26,000/mo | $73,000/mo |
| 5B | $7,999/mo | Contact sales | Contact sales | Contact sales |
| 10B | $14,999/mo | Contact sales | Contact sales | Contact sales |

Competitor prices from public pricing calculators (768d, ~100 QPS). Qdrant = dedicated cluster; Pinecone = serverless read units; Weaviate = per-dimension pricing. MVS includes scale-to-zero — idle namespaces cost only storage.

First 1M vectors are free on the Starter tier. Growth tier minimum ($50/mo) is applied toward usage — if your usage exceeds $50, you just pay usage. Enterprise gets volume discounts.

## Systems tested

| Engine | Index type | Storage model | Hybrid search |
|--------|-----------|--------------|---------------|
| [MVS](https://mixpeek.com/mvs) | LIRE partitions + PQ | BYO object storage (S3/GCS/B2) | tantivy BM25 |
| [Qdrant](https://qdrant.tech) | HNSW | On-disk + RAM | Payload indexes |
| [Milvus](https://milvus.io) | IVF / HNSW | Distributed, GPU support | Sparse vectors |
| [Weaviate](https://weaviate.io) | HNSW | In-memory + disk | BM25 modules |
| [Chroma](https://trychroma.com) | HNSW (hnswlib) | In-memory | Metadata filtering |
| [LanceDB](https://lancedb.com) | IVF-PQ | Lance columnar format | Full-text search |
| [pgvector](https://github.com/pgvector/pgvector) | IVFFlat / HNSW | Postgres-native | SQL + tsvector |

Each engine runs in Docker with vendor-recommended configurations. PRs with improved configs are welcome.

## Scenarios

| # | Scenario | What it tests |
|---|----------|--------------|
| 1 | Steady-state search | Recall vs QPS at scale |
| 2 | Streaming ingest | p99 latency during continuous insert |
| 3 | Memory-constrained | 1B vectors in 32 GB RAM |
| 4 | 12-month TCO | Cost per million queries over time |
| 5 | Hybrid search | BM25 + dense vector NDCG@10 |
| 6 | Filtered search | Recall at varying filter selectivity |
| 7 | Cold-start recovery | Time to serve after process restart |

Results above are from Scenario 1 (steady-state). Other scenarios are in progress.

## Datasets

| Dataset | Vectors | Dimensions | Source |
|---------|--------:|----------:|--------|
| Cohere-1B-768 | 1,000,000,000 | 768 | Wikipedia via Cohere embed-v3 |
| Cohere-10M-768 | 10,000,000 | 768 | Wikipedia subset |
| DEEP1B | 1,000,000,000 | 96 | Web images |
| BEIR (multiple) | Varies | 768 | IR evaluation |
| YFCC-10M | 10,000,000 | 192 | Flickr + metadata |

## Quick start

```bash
git clone https://github.com/mixpeek/mvs-benchmark.git
cd mvs-benchmark

# Run one scenario against one engine (50K vectors, quick test)
python benchmark/scripts/run.py \
  --scenario steady-state \
  --engine mvs \
  --dataset cohere-10m-768 \
  --output results/

# Run all scenarios against all engines (~24h, requires 64+ GB RAM)
python benchmark/scripts/run_all.py --scale 1b

# Generate results site
python benchmark/site/generate.py --results results/ --output docs/

# Validate result JSON before publishing or comparing runs
python benchmark/scripts/validate_results.py results/
```

## Reproducing results

Every published result includes the exact Docker image SHA, engine config, hardware spec, and raw JSON. To reproduce:

```bash
python benchmark/scripts/reproduce.py --result results/2026-04/steady-state-mvs.json
```

## Contributing

We welcome PRs with optimized engine configurations:

```bash
cp benchmark/engines/mvs/config.yaml benchmark/engines/your-engine/config.yaml
# Edit, test locally, submit PR
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details. See [METHODOLOGY.md](benchmark/METHODOLOGY.md) for measurement methodology.

## License

Apache 2.0
