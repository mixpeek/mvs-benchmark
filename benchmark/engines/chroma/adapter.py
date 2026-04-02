"""Chroma engine adapter for benchmarking."""

import time
import numpy as np
import chromadb

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class ChromaAdapter(EngineAdapter):

    def __init__(self, host: str = "localhost", port: int = 8100):
        self._host = host
        self._port = port
        self._client = None
        self._collection = None
        self._collection_name: str = ""

    @property
    def name(self) -> str:
        return "chroma"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._collection_name = collection_name
        self._client = chromadb.HttpClient(host=self._host, port=self._port)

        # Delete if exists
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass

        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 200},
        )

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        self._collection.add(
            ids=ids,
            embeddings=vectors.tolist(),
        )

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        start = time.perf_counter()
        results = self._collection.query(
            query_embeddings=[vector.tolist()],
            n_results=top_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result_ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results.get("distances") else []

        return SearchMetrics(
            results=[
                SearchResult(id=rid, score=1.0 - d if distances else 0.0)
                for rid, d in zip(result_ids, distances or [0] * len(result_ids))
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        return self._collection.count() if self._collection else 0

    def wait_for_index(self) -> None:
        time.sleep(1)

    def teardown(self) -> None:
        if self._client and self._collection_name:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
