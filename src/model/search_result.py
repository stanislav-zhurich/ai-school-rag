from dataclasses import dataclass
from typing import Any

@dataclass
class SearchResult:
    """A single result returned by :meth:`ChromaDBStore.search`."""
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float