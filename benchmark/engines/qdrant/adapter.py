"""Qdrant engine adapter for benchmarking."""

import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class QdrantAdapter(EngineAdapter):
    """Adapter for Qdrant vector database."""

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        self._url = url
        self._api_key = api_key
        self._client: QdrantClient | None = None
        self._collection_name: str = ""

    @property
    def name(self) -> str:
        return "qdrant"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._collection_name = collection_name
        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=120)

        # Delete if exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name in collections:
            self._client.delete_collection(collection_name)

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=Distance.COSINE,
            ),
            on_disk_payload=kwargs.get("on_disk_payload", False),
        )

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        assert self._client is not None
        points = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
            # Qdrant requires unsigned int or UUID for point IDs
            point_id = int(id_) if id_.isdigit() else id_
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        assert self._client is not None
        start = time.perf_counter()
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=vector.tolist(),
            limit=top_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return SearchMetrics(
            results=[
                SearchResult(id=str(r.id), score=r.score, payload=r.payload)
                for r in results.points
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        assert self._client is not None
        info = self._client.get_collection(self._collection_name)
        return info.points_count or 0

    def wait_for_index(self) -> None:
        """Wait for Qdrant to finish indexing (status = green)."""
        assert self._client is not None
        while True:
            info = self._client.get_collection(self._collection_name)
            if info.status.value == "green":
                break
            time.sleep(1)

    def teardown(self) -> None:
        if self._client:
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass
            self._client.close()
