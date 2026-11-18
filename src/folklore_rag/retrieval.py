"""Embedding + Qdrant retrieval (in-memory)."""
from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import EMBEDDING_MODELS
from .corpus import Story

log = logging.getLogger(__name__)


class Retriever:
    """Indexes the corpus with one or more embedding models and serves top-k search."""

    def __init__(self, corpus: list[Story], keys: list[str] | None = None):
        self.corpus = corpus
        self.qdrant = QdrantClient(":memory:")
        self._models: dict = {}
        for key in keys or list(EMBEDDING_MODELS):
            self.index(key)

    def _model(self, key: str):
        if key not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[key] = SentenceTransformer(EMBEDDING_MODELS[key])
        return self._models[key]

    def embed(self, key: str, texts: list[str], kind: str = "passage") -> list[list[float]]:
        if key == "e5":  # e5 models expect a "query: " / "passage: " prefix
            texts = [f"{kind}: {t}" for t in texts]
        return self._model(key).encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def index(self, key: str) -> None:
        name = f"stories_{key}"
        if self.qdrant.collection_exists(name):
            self.qdrant.delete_collection(name)
        vecs = self.embed(key, [f"{s['title']}. {s['text']}" for s in self.corpus])
        self.qdrant.create_collection(name, vectors_config=VectorParams(size=len(vecs[0]), distance=Distance.COSINE))
        self.qdrant.upsert(name, points=[PointStruct(id=i, vector=v, payload=dict(self.corpus[i])) for i, v in enumerate(vecs)])
        log.info("Indexed %d stories with %s", len(self.corpus), key)

    def search(self, key: str, query: str, k: int = 3):
        vec = self.embed(key, [query], "query")[0]
        return self.qdrant.query_points(collection_name=f"stories_{key}", query=vec, limit=k).points

    def nearest(self, key: str, text: str) -> tuple[str, float]:
        """Return (title, cosine similarity) of the closest corpus story to ``text``."""
        vec = self.embed(key, [text], "passage")[0]
        top = self.qdrant.query_points(collection_name=f"stories_{key}", query=vec, limit=1).points[0]
        return top.payload["title"], float(top.score)
