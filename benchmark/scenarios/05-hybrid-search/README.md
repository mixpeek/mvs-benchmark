# Scenario 5: Hybrid Search Quality

**What it tests:** Combined BM25 + dense vector search quality on standard IR benchmarks.

Production search systems need both keyword precision and semantic recall. This scenario measures how well each engine combines sparse and dense retrieval.

## Setup

- **Dataset:** BEIR benchmark suite (MS MARCO, NQ, FiQA, SciFact, NFCorpus, etc.)
- **Dense embeddings:** Cohere embed-v3 (768 dimensions)
- **Sparse signal:** BM25 or SPLADE, depending on engine capability
- **Fusion:** Reciprocal rank fusion (RRF) with k=60

## Systems and their hybrid approach

| System | Dense | Sparse | Fusion |
|---|---|---|---|
| MVS | LIRE partitions | tantivy BM25 (native) | Server-side RRF |
| Qdrant | HNSW | Sparse vectors (SPLADE) | Query-time fusion |
| Milvus | HNSW/IVF | BM25 (2.4+) | RangeSearch + rerank |
| Weaviate | HNSW | BM25 (built-in) | Hybrid alpha parameter |
| pgvector | HNSW/IVFFlat | tsvector + GIN (Postgres FTS) | Application-side |

## Metrics

| Metric | How measured |
|---|---|
| NDCG@10 | Primary quality metric, per BEIR protocol |
| MAP@100 | Mean average precision |
| Recall@100 | Fraction of relevant docs in top 100 |
| Latency | P50/P95/P99 for hybrid queries |
| Dense-only baseline | NDCG@10 without sparse signal |

## Expected outcome

Engines with native, high-quality BM25 (tantivy, Postgres tsvector) will outperform those using SPLADE or approximate sparse methods on keyword-heavy queries. MVS's tantivy integration should show strong gains on datasets like FiQA and SciFact where exact term matching matters.

## Run

```bash
python benchmark/scripts/run.py \
  --scenario hybrid-search \
  --engine all \
  --dataset beir
```
