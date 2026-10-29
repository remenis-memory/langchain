"""Remenis zero-infra local security governance vector store."""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

try:
    from remenis import MemoryEngine, RingContextGuard
except ImportError:
    raise ImportError(
        "Could not import remenis python package. "
        "Please install it with `pip install remenis-memory`."
    )

from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RemenisVectorStore(VectorStore):
    """Remenis zero-infra, air-gapped memory and security governance store."""

    def __init__(
        self,
        collection_name: str = "langchain_remenis",
        ring_level: int = 1,
        **kwargs: Any,
    ) -> None:
        self.guard = RingContextGuard(ring_level=ring_level)
        self.engine = MemoryEngine(guard=self.guard)
        self.collection_name = collection_name

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids = []
        metadatas = metadatas or [{} for _ in texts]
        for text, meta in zip(texts, metadatas):
            clean_context = self.engine.process_input(text)
            record_id = self.engine.store_memory(content=clean_context, metadata=meta)
            ids.append(str(record_id))
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self.engine.similarity_search(query=query, limit=k)
        docs = []
        for res in results:
            doc = Document(
                page_content=res.get("content", ""),
                metadata=res.get("metadata", {})
            )
            docs.append(doc)
        return docs

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any = None,
        metadatas: Optional[List[dict[str, Any]]] = None,
        collection_name: str = "langchain_remenis",
        **kwargs: Any,
    ) -> RemenisVectorStore:
        store = cls(collection_name=collection_name, **kwargs)
        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store
