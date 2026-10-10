"""Story 1.1 — dense embedding service (multilingual-e5-small)."""

from __future__ import annotations

from mc.config import DENSE_DIM, DENSE_MODEL_ID
from mc.embeddings import DenseEmbedder, embed_passage, embed_query


def test_embed_passage_returns_dense_dim_vector():
    vec = embed_passage("o bitcoin é uma bolha")[0]
    assert len(vec) == DENSE_DIM == 384
    assert all(isinstance(x, float) for x in vec)


def test_query_and_passage_prefixes_differ():
    raw = "regulação de cripto no brasil"
    q = embed_query(raw)
    p = embed_passage(raw)[0]
    assert q != p, "query prefix must be applied — vectors must differ"


def test_batch_preserves_order():
    texts = ["primeiro texto", "segundo texto bem diferente", "terceiro"]
    out = embed_passage(texts)
    assert len(out) == 3
    single = [embed_passage(t)[0] for t in texts]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    for i in range(3):
        # out[i] must correspond to texts[i]: near-identical to the single
        # encode of the same text (float batching may differ in last digits).
        assert dot(out[i], single[i]) > 0.999, (
            f"out[{i}] does not correspond to texts[{i}]"
        )
        for j in range(3):
            if i != j:
                assert dot(out[i], single[j]) < dot(out[i], single[i])


def test_deterministic():
    a = embed_passage("determinismo do modelo")[0]
    b = embed_passage("determinismo do modelo")[0]
    assert a == b


def test_model_identity_exposed():
    assert DenseEmbedder.model_id == DENSE_MODEL_ID == "intfloat/multilingual-e5-small"
    assert DenseEmbedder.dim == 384


def test_model_fingerprint_stable():
    from mc.embeddings import model_fingerprint

    fp = model_fingerprint()
    assert fp["model"] == DENSE_MODEL_ID
    assert len(fp["hash"]) >= 8
    assert model_fingerprint() == fp
