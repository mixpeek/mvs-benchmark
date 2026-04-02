"""pgvector engine adapter for benchmarking."""

import time
import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class PgvectorAdapter(EngineAdapter):

    def __init__(self, host: str = "localhost", port: int = 5433,
                 dbname: str = "bench", user: str = "postgres", password: str = "bench"):
        self._dsn = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        self._conn = None
        self._table_name: str = ""
        self._dimensions: int = 768

    @property
    def name(self) -> str:
        return "pgvector"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._table_name = collection_name.replace("-", "_")
        self._dimensions = dimensions
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True

        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"DROP TABLE IF EXISTS {self._table_name};")
            cur.execute(f"""
                CREATE TABLE {self._table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dimensions})
                );
            """)

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        with self._conn.cursor() as cur:
            values = [(id_, vec.tolist()) for id_, vec in zip(ids, vectors)]
            execute_values(
                cur,
                f"INSERT INTO {self._table_name} (id, embedding) VALUES %s",
                values,
                template="(%s, %s::vector)",
            )

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        start = time.perf_counter()
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT id, 1 - (embedding <=> %s::vector) as score "
                f"FROM {self._table_name} "
                f"ORDER BY embedding <=> %s::vector "
                f"LIMIT %s",
                (vector.tolist(), vector.tolist(), top_k),
            )
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000

        return SearchMetrics(
            results=[SearchResult(id=str(r[0]), score=float(r[1])) for r in rows],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table_name}")
            return cur.fetchone()[0]

    def wait_for_index(self) -> None:
        """Build HNSW index."""
        with self._conn.cursor() as cur:
            cur.execute(f"""
                CREATE INDEX ON {self._table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 200);
            """)
            # Set search ef
            cur.execute("SET hnsw.ef_search = 128;")

    def teardown(self) -> None:
        if self._conn:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {self._table_name};")
            except Exception:
                pass
            self._conn.close()
