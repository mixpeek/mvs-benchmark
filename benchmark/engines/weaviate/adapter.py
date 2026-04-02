"""Weaviate engine adapter for benchmarking."""

import time
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class WeaviateAdapter(EngineAdapter):

    def __init__(self, url: str = "http://localhost:8079"):
        self._url = url
        self._client = None
        self._collection_name: str = ""

    @property
    def name(self) -> str:
        return "weaviate"

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        self._collection_name = collection_name.replace("-", "_").capitalize()
        self._client = weaviate.connect_to_local(
            host=self._url.replace("http://", "").split(":")[0],
            port=int(self._url.split(":")[-1]),
            grpc_port=50052,
            skip_init_checks=True,
        )

        if self._client.collections.exists(self._collection_name):
            self._client.collections.delete(self._collection_name)

        self._client.collections.create(
            name=self._collection_name,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=200,
                max_connections=16,
                ef=128,
            ),
            properties=[
                Property(name="item_id", data_type=DataType.TEXT),
            ],
        )

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        collection = self._client.collections.get(self._collection_name)
        with collection.batch.dynamic() as batch:
            for i, (id_, vec) in enumerate(zip(ids, vectors)):
                batch.add_object(
                    properties={"item_id": id_},
                    vector=vec.tolist(),
                )

    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        collection = self._client.collections.get(self._collection_name)
        start = time.perf_counter()
        result = collection.query.near_vector(
            near_vector=vector.tolist(),
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return SearchMetrics(
            results=[
                SearchResult(
                    id=str(obj.properties.get("item_id", "")),
                    score=1.0 - (obj.metadata.distance or 0),
                )
                for obj in result.objects
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        collection = self._client.collections.get(self._collection_name)
        resp = collection.aggregate.over_all(total_count=True)
        return resp.total_count or 0

    def wait_for_index(self) -> None:
        time.sleep(2)

    def teardown(self) -> None:
        if self._client:
            try:
                self._client.collections.delete(self._collection_name)
            except Exception:
                pass
            self._client.close()
