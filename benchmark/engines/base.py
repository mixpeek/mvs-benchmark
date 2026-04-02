"""Base class for engine adapters."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    """A single search result with ID and score."""
    id: str
    score: float
    payload: dict | None = None


@dataclass
class SearchMetrics:
    """Metrics from a single search query."""
    results: list[SearchResult]
    latency_ms: float


class EngineAdapter(ABC):
    """Base class for all engine adapters.

    Each adapter knows how to:
    - Create a collection/namespace
    - Insert vectors
    - Search vectors
    - Report health/readiness
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def setup(self, collection_name: str, dimensions: int, **kwargs) -> None:
        """Create collection and prepare for inserts."""
        ...

    @abstractmethod
    def insert_batch(self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None) -> None:
        """Insert a batch of vectors."""
        ...

    @abstractmethod
    def search(self, vector: np.ndarray, top_k: int = 10) -> SearchMetrics:
        """Search for nearest neighbors. Returns results + latency."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the collection."""
        ...

    @abstractmethod
    def wait_for_index(self) -> None:
        """Block until indexing is complete."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Clean up the collection."""
        ...

    def health_check(self) -> bool:
        """Return True if engine is reachable."""
        try:
            self.count()
            return True
        except Exception:
            return False
