"""Embedding layer: dense (multilingual-e5-small) + sparse (fastembed BM25).

Stories 1.1/1.2. Model identity is exposed for NFR-6 fingerprinting so
collection mismatch is rejected loudly (Story 1.3).
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

from mc.config import (
    DENSE_DIM,
    DENSE_MODEL_ID,
    SPARSE_LANGUAGE,
    SPARSE_MODEL_ID,
)

# NFR-1: cap CPU threads so one embedding call can't starve a 2-4 vCPU VPS.
TORCH_NUM_THREADS = 2
torch_lock = threading.Lock()

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


@dataclass(frozen=True)
class SparseVector:
    """Format-compatible with Qdrant sparse points."""

    indices: tuple[int, ...]
    values: tuple[float, ...]


class _DenseEmbedder:
    """Singleton wrapper around sentence-transformers with fixed seed."""

    model_id: str = DENSE_MODEL_ID
    dim: int = DENSE_DIM
    _instance: object | None = None
    _lock = threading.Lock()

    @classmethod
    def _model(cls):
        with cls._lock:
            if cls._instance is None:
                import torch

                torch.set_num_threads(TORCH_NUM_THREADS)
                torch.manual_seed(42)
                from sentence_transformers import SentenceTransformer

                cls._instance = SentenceTransformer(DENSE_MODEL_ID)
            return cls._instance

    @classmethod
    def embed_passage(cls, texts: list[str] | str) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        prefixed = [f"{PASSAGE_PREFIX}{t}" for t in texts]
        with torch_lock:
            out = cls._model().encode(prefixed, normalize_embeddings=True)
        return [row.tolist() for row in out]

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        with torch_lock:
            out = cls._model().encode(
                [f"{QUERY_PREFIX}{text}"], normalize_embeddings=True
            )
        return out[0].tolist()


# Module-level facade (DoD names these directly).
DenseEmbedder = _DenseEmbedder


def embed_passage(texts: list[str]) -> list[list[float]]:
    return _DenseEmbedder.embed_passage(texts)


def embed_query(text: str) -> list[float]:
    return _DenseEmbedder.embed_query(text)


def model_fingerprint() -> dict:
    """Identity of the dense model, stored alongside vectors (NFR-6)."""
    return {
        "model": DENSE_MODEL_ID,
        "dim": DENSE_DIM,
        "hash": _fingerprint_hash(f"{DENSE_MODEL_ID}:{DENSE_DIM}"),
    }


def sparse_fingerprint() -> dict:
    return {
        "model": SPARSE_MODEL_ID,
        "language": SPARSE_LANGUAGE,
        "hash": _fingerprint_hash(f"{SPARSE_MODEL_ID}:{SPARSE_LANGUAGE}"),
    }


def _fingerprint_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class _SparseEncoder:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def _model(cls):
        with cls._lock:
            if cls._instance is None:
                from fastembed import SparseTextEmbedding

                cls._instance = SparseTextEmbedding(
                    model_name=SPARSE_MODEL_ID,
                )
            return cls._instance

    @classmethod
    def embed(cls, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        model = cls._model()
        out = []
        for vec in model.embed(texts, language=SPARSE_LANGUAGE):
            out.append(
                SparseVector(
                    indices=tuple(int(i) for i in vec.indices),
                    values=tuple(float(v) for v in vec.values),
                )
            )
        return out


def embed_sparse(texts: list[str] | str) -> list[SparseVector]:
    if isinstance(texts, str):
        texts = [texts]
    return _SparseEncoder.embed(texts)
