"""Story 1.2 — sparse BM25 encoder via fastembed."""

from __future__ import annotations

from mc.config import SPARSE_MODEL_ID
from mc.embeddings import embed_sparse, sparse_fingerprint


def test_same_text_identical_sparse_vector():
    a = embed_sparse(["mercado cripto em alta"])
    b = embed_sparse(["mercado cripto em alta"])
    assert a[0] == b[0]


def test_different_texts_different_token_counts():
    a = embed_sparse(["sol"])[0]
    b = embed_sparse(
        ["como proteger o patrimônio com bitcoins e altcoins hoje"]
    )[0]
    assert len(a.indices) != len(b.indices)


def test_vector_format_valid_for_qdrant():
    vec = embed_sparse(["regulação das stablecoins"])[0]
    assert list(vec.indices) == sorted(vec.indices), "indices must be unique/ascending"
    assert len(vec.indices) == len(vec.values)
    assert all(isinstance(i, int) for i in vec.indices)


def test_sparse_fingerprint():
    fp = sparse_fingerprint()
    assert fp["model"] == SPARSE_MODEL_ID
    assert sparse_fingerprint() == fp
