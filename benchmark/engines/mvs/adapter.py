"""MVS (Mixpeek Vector Store) engine adapter for benchmarking.

Uses the Mixpeek production API to test the full MVS stack:
namespace → bucket → collection → retriever → search.
"""

import time
import httpx
import numpy as np

from benchmark.engines.base import EngineAdapter, SearchMetrics, SearchResult


class MVSAdapter(EngineAdapter):
    """Adapter for MVS via the Mixpeek API.

    This adapter benchmarks MVS through the production Mixpeek API,
    which includes the full stack: API layer, feature extraction,
    LIRE partitions, and BYO storage.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.mixpeek.com",
        timeout: float = 60.0,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._namespace_id: str | None = None
        self._bucket_id: str | None = None
        self._collection_id: str | None = None
        self._retriever_id: str | None = None
        self._dimensions: int = 768

    @property
    def name(self) -> str:
        return "mvs"

    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self._api_key}"}
        if self._namespace_id:
            h["X-Namespace"] = self._namespace_id
        return h

    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        """Set up the full Mixpeek pipeline: namespace → bucket → collection → retriever."""
        self._dimensions = dimensions

        # If namespace_id is provided, reuse existing infrastructure
        if kwargs.get("namespace_id"):
            self._namespace_id = kwargs["namespace_id"]
            self._retriever_id = kwargs.get("retriever_id")
            self._collection_id = kwargs.get("collection_id")
            self._bucket_id = kwargs.get("bucket_id")
            return

        # Create namespace
        resp = self._client.post(
            f"{self._base_url}/v1/namespaces",
            headers=self._headers(),
            json={
                "namespace_name": f"bench-{collection_name}",
                "feature_extractors": [
                    {"feature_extractor_name": "text_extractor", "version": "v1"},
                ],
            },
        )
        resp.raise_for_status()
        self._namespace_id = resp.json()["namespace_id"]

        # Create bucket
        resp = self._client.post(
            f"{self._base_url}/v1/buckets",
            headers=self._headers(),
            json={
                "bucket_name": f"bench-{collection_name}-data",
                "bucket_schema": {
                    "properties": {
                        "text": {"type": "string"},
                        "item_id": {"type": "string"},
                    }
                },
            },
        )
        resp.raise_for_status()
        self._bucket_id = resp.json()["bucket_id"]

        # Create collection
        resp = self._client.post(
            f"{self._base_url}/v1/collections",
            headers=self._headers(),
            json={
                "collection_name": f"bench-{collection_name}-col",
                "source": {
                    "type": "bucket",
                    "bucket_ids": [self._bucket_id],
                },
                "feature_extractor": {
                    "feature_extractor_name": "text_extractor",
                    "version": "v1",
                    "input_mappings": {"text": "text"},
                    "parameters": {},
                    "field_passthrough": ["item_id"],
                },
            },
        )
        resp.raise_for_status()
        self._collection_id = resp.json()["collection_id"]

        # Create retriever
        resp = self._client.post(
            f"{self._base_url}/v1/retrievers",
            headers=self._headers(),
            json={
                "retriever_name": f"bench-{collection_name}-ret",
                "collection_identifiers": [self._collection_id],
                "stages": [
                    {
                        "stage_name": "vector_search",
                        "stage_type": "filter",
                        "config": {
                            "stage_id": "feature_search",
                            "parameters": {
                                "searches": [
                                    {
                                        "feature_uri": "mixpeek://text_extractor@v1/all_minilm_l6_v2_v1",
                                        "query": {
                                            "input_mode": "text",
                                            "text": "{{INPUT.query}}",
                                        },
                                        "top_k": 10,
                                    }
                                ],
                                "final_top_k": 10,
                                "fusion": "rrf",
                                "collection_identifiers": [self._collection_id],
                            },
                        },
                    },
                ],
                "input_schema": {
                    "query": {
                        "type": "string",
                        "description": "Search query text",
                    },
                },
            },
        )
        resp.raise_for_status()
        self._retriever_id = resp.json()["retriever"]["retriever_id"]

    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        """Insert data through the Mixpeek bucket API.

        Note: MVS ingestion goes through the full pipeline (bucket → collection → processing).
        For benchmarking insert throughput, we insert objects into the bucket and wait for
        processing to complete.
        """
        assert self._bucket_id is not None

        for i, id_ in enumerate(ids):
            payload = payloads[i] if payloads else {}
            text = payload.get("text", f"benchmark item {id_}")

            resp = self._client.post(
                f"{self._base_url}/v1/buckets/{self._bucket_id}/objects",
                headers=self._headers(),
                json={
                    "item_id": id_,
                    "text": text,
                    "blobs": [
                        {
                            "property": "text",
                            "type": "text",
                            "data": text,
                        }
                    ],
                },
            )
            resp.raise_for_status()

    def search(self, vector: np.ndarray, top_k: int = 10, query_text: str | None = None) -> SearchMetrics:
        """Search via Mixpeek retriever.

        Note: MVS search goes through the full retriever pipeline, which includes
        embedding generation. For raw vector search benchmarking, this measures
        end-to-end latency (embedding + search + ranking).
        """
        assert self._retriever_id is not None

        # Use query text if provided, otherwise generate a synthetic query
        text = query_text or "benchmark search query"

        start = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/v1/retrievers/{self._retriever_id}/execute",
            headers=self._headers(),
            json={"inputs": {"query": text}},
        )
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()

        data = resp.json()
        documents = data.get("documents", [])

        return SearchMetrics(
            results=[
                SearchResult(
                    id=str(doc.get("document_id", "")),
                    score=doc.get("score", 0.0),
                    payload=doc,
                )
                for doc in documents
            ],
            latency_ms=latency_ms,
        )

    def count(self) -> int:
        """Get document count from the collection."""
        if not self._collection_id:
            return 0
        resp = self._client.get(
            f"{self._base_url}/v1/collections/{self._collection_id}",
            headers=self._headers(),
        )
        if resp.status_code == 200:
            return resp.json().get("document_count", 0)
        return 0

    def wait_for_index(self) -> None:
        """Wait for batch processing to complete."""
        # Poll until no active batches
        for _ in range(300):  # 5 min max
            time.sleep(1)
            try:
                resp = self._client.get(
                    f"{self._base_url}/v1/collections/{self._collection_id}/batches",
                    headers=self._headers(),
                    params={"status": "processing"},
                )
                if resp.status_code == 200:
                    batches = resp.json()
                    active = [b for b in batches if b.get("status") in ("processing", "pending")]
                    if not active:
                        return
            except Exception:
                pass

    def teardown(self) -> None:
        """Delete the namespace (cascades to bucket, collection, retriever)."""
        if self._namespace_id:
            try:
                self._client.delete(
                    f"{self._base_url}/v1/namespaces/{self._namespace_id}",
                    headers=self._headers(),
                )
            except Exception:
                pass
        self._client.close()
