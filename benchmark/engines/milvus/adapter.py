"""Milvus engine adapter for benchmarking."""

import time
import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class MilvusAdapter(EngineAdapter):

    def __init__(self, host: str = "localhost", port: int = 19530):
        self._host = host
        self._port = port
        self._collection: Collection | None = None
        self._collection_name: str = ""
        self._dimensions: int = 768

    @property
    def name(self) -> str:
        return "milvus"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._collection_name = collection_name.replace("-", "_")
        self._dimensions = dimensions

        connections.connect(host=self._host, port=self._port)

        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)

        schema = CollectionSchema(fields=[
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimensions),
        ])

        self._collection = Collection(name=self._collection_name, schema=schema)

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        self._collection.insert([ids, vectors.tolist()])

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        start = time.perf_counter()
        results = self._collection.search(
            data=[vector.tolist()],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        hits = results[0] if results else []
        return SearchMetrics(
            results=[
                SearchResult(id=str(hit.id), score=hit.score)
                for hit in hits
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        if self._collection:
            self._collection.flush()
            return self._collection.num_entities
        return 0

    def wait_for_index(self) -> None:
        """Build HNSW index and load collection into memory."""
        self._collection.flush()
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        self._collection.load()
        # Wait for load to complete
        utility.wait_for_loading_complete(self._collection_name)

    def teardown(self) -> None:
        if self._collection:
            try:
                self._collection.release()
                utility.drop_collection(self._collection_name)
            except Exception:
                pass
        try:
            connections.disconnect("default")
        except Exception:
            pass
