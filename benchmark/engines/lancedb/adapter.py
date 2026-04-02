"""LanceDB engine adapter for benchmarking.

LanceDB is embedded (no server), so it runs in-process.
"""

import shutil
import time
import numpy as np
import lancedb
import pyarrow as pa

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class LanceDBAdapter(EngineAdapter):

    def __init__(self, db_path: str = "/tmp/bench-lancedb"):
        self._db_path = db_path
        self._db = None
        self._table = None
        self._table_name: str = ""

    @property
    def name(self) -> str:
        return "lancedb"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._table_name = collection_name.replace("-", "_")
        # Clean up any existing DB
        shutil.rmtree(self._db_path, ignore_errors=True)
        self._db = lancedb.connect(self._db_path)

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        data = pa.table({
            "id": ids,
            "vector": [vec.tolist() for vec in vectors],
        })

        if self._table is None:
            self._table = self._db.create_table(self._table_name, data)
        else:
            self._table.add(data)

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        start = time.perf_counter()
        results = (
            self._table.search(vector.tolist())
            .metric("cosine")
            .limit(top_k)
            .to_list()
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return SearchMetrics(
            results=[
                SearchResult(
                    id=str(r["id"]),
                    score=1.0 - r.get("_distance", 0),
                )
                for r in results
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        return len(self._table) if self._table else 0

    def wait_for_index(self) -> None:
        """Create IVF-PQ index for LanceDB."""
        if self._table and len(self._table) >= 256:
            try:
                self._table.create_index(
                    metric="cosine",
                    num_partitions=min(256, len(self._table) // 50),
                    num_sub_vectors=min(96, 768 // 8),
                )
            except Exception:
                pass  # Falls back to flat search
        time.sleep(1)

    def teardown(self) -> None:
        shutil.rmtree(self._db_path, ignore_errors=True)
