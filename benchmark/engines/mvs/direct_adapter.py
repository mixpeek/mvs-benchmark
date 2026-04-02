"""MVS direct vector search adapter — bypasses embedding, tests pure search.

MVS currently uses Qdrant as its hot serving layer. This adapter inserts
and searches vectors through the same Qdrant path that MVS uses in production,
giving an apples-to-apples comparison with other vector databases.

For the LIRE partition path (MVS's own index), see lire_adapter.py (future).
"""

import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class MVSDirectAdapter(EngineAdapter):
    """Test MVS's vector search layer directly with pre-computed vectors.

    This bypasses the Mixpeek API and embedding generation, talking directly
    to the Qdrant instance that MVS uses as its hot serving layer.
    Same insert/search path as production MVS reads.
    """

    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None):
        self._url = url
        self._api_key = api_key
        self._client: QdrantClient | None = None
        self._collection_name: str = ""

    @property
    def name(self) -> str:
        return "mvs"

    def health_check(self) -> bool:
        try:
            c = QdrantClient(url=self._url, api_key=self._api_key, timeout=5)
            c.get_collections()
            c.close()
            return True
        except Exception:
            return False

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._collection_name = collection_name
        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=120)

        # Delete if exists
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name in collections:
            self._client.delete_collection(collection_name)

        # Create with same config MVS uses in production
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimensions,
                distance=Distance.COSINE,
            ),
        )

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        assert self._client is not None
        points = []
        for i, (id_, vec) in enumerate(zip(ids, vectors)):
            payload = payloads[i] if payloads else {}
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
