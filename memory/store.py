"""Vector memory store for the Memory Curator (RAG).

Two backends:

* :class:`InMemoryMemoryStore` — deterministic, dependency-free, hashing-based
  vectorizer used for tests and offline evaluation.
* :class:`ChromaMemoryStore` — production ChromaDB backend.

The store is content-addressable: the Memory Curator writes a *summary* of each
deployment (problem class, provider, outcome) and later retrieves the most
similar past deployments to condition the DevOps and FinOps agents.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from cloud_orchestra.schemas import MemoryEntry


class _HashingVectorizer:
    """Character n-gram hashing vectorizer (deterministic, no external deps)."""

    DIM = 256

    def __init__(self, n: int = 3) -> None:
        self._n = n

    def transform(self, text: str) -> list[float]:
        text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        tokens = re.findall(r"\w+", text)
        vec = [0.0] * self.DIM
        for token in tokens:
            padded = f" {token} "
            grams = [padded[i : i + self._n] for i in range(max(1, len(padded) - self._n + 1))]
            for gram in grams:
                idx = int(hashlib.md5(gram.encode()).hexdigest(), 16) % self.DIM
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    @staticmethod
    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))


class MemoryStore:
    async def add(self, entry: MemoryEntry, text: str) -> None:
        raise NotImplementedError

    async def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._vectorizer = _HashingVectorizer()
        self._entries: list[MemoryEntry] = []
        self._vectors: list[list[float]] = []

    async def add(self, entry: MemoryEntry, text: str) -> None:
        self._entries.append(entry)
        self._vectors.append(self._vectorizer.transform(text))

    async def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        if not self._entries:
            return []
        qv = self._vectorizer.transform(query)
        scored = [
            (entry, self._vectorizer.cosine(qv, vec))
            for entry, vec in zip(self._entries, self._vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]


class ChromaMemoryStore(MemoryStore):
    def __init__(self, persist_dir: str, collection: str = "deployments") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection
        self._client: Any = None
        self._collection: Any = None
        self._entries: dict[str, MemoryEntry] = {}

    def _ensure(self) -> None:
        if self._client is not None:
            return
        import chromadb  # guarded import

        self._client = chromadb.PersistentClient(path=self._persist_dir)
        self._collection = self._client.get_or_create_collection(self._collection_name)

    async def add(self, entry: MemoryEntry, text: str) -> None:
        self._ensure()
        self._entries[str(entry.id)] = entry
        self._collection.add(
            ids=[str(entry.id)],
            documents=[text],
            metadatas=[{"problem_class": entry.problem_class, "provider": entry.provider.value}],
        )

    async def search(self, query: str, top_k: int = 5) -> list[tuple[MemoryEntry, float]]:
        self._ensure()
        result = self._collection.query(query_texts=[query], n_results=top_k)
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        out: list[tuple[MemoryEntry, float]] = []
        for idx, doc_id in enumerate(ids):
            entry = self._entries.get(doc_id)
            if entry is None:
                continue
            distance = distances[idx] if idx < len(distances) else 0.0
            out.append((entry, 1.0 - float(distance)))
        return out

    async def close(self) -> None:  # pragma: no cover - trivial
        self._client = None


def build_memory_store(provider: str, persist_dir: str = "./chroma_data") -> MemoryStore:
    if provider == "chroma":
        return ChromaMemoryStore(persist_dir)
    return InMemoryMemoryStore()
