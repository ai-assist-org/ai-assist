"""Embedding model wrapper for semantic search in the knowledge graph"""

import logging
import os
import threading
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    _instance: EmbeddingModel | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model: Any = None
        self._dimensions: int | None = None
        self._model_name = os.getenv("AI_ASSIST_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    @classmethod
    def get(cls) -> EmbeddingModel:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def preload(cls) -> None:
        """Trigger model loading in a background thread."""
        thread = threading.Thread(target=cls.get()._load, daemon=True)
        thread.start()

    def _load(self) -> None:
        if self._model is not None:
            return
        from fastembed import TextEmbedding

        self._model = TextEmbedding(self._model_name)
        self._dimensions = self._model.embedding_size
        logging.info("Embedding model loaded (%s, %d dims)", self._model_name, self._dimensions)

    def encode(self, texts: list[str]) -> np.ndarray:
        self._load()
        embeddings = list(self._model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

    def encode_one(self, text: str) -> bytes:
        arr = self.encode([text])[0]
        return arr.astype(np.float32).tobytes()

    @property
    def dimensions(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        self._load()
        assert self._dimensions is not None
        return self._dimensions
