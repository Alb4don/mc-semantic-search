"""Story 1.3 — Qdrant collection manager.

Integration tests against a local qdrant container (docker compose up -d qdrant).
Each test uses a unique collection name so runs don't clash.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from mc.config import DENSE_DIM, DENSE_MODEL_ID
from mc.embeddings import model_fingerprint
from mc.qdrant import (
    COLLECTION_META_POINT_ID,
    ModelMismatchError,
    ensure_collection,
    get_collection,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=5)
    try:
        client.get_collections()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"local qdrant unreachable: {exc}")
    return client


@pytest.fixture
def collection_name(qdrant: QdrantClient) -> str:
    name = f"mc-test-{uuid.uuid4().hex[:8]}"
    yield name
    qdrant.delete_collection(name)


def test_create_sets_named_vectors_and_payload_indexes(qdrant, collection_name):
    ensure_collection(qdrant, collection_name)

    info = get_collection(qdrant, collection_name)
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors
    # Named vectors: dense (384, cosine) + sparse (bm25).
    assert "dense" in vectors
    assert vectors["dense"].size == DENSE_DIM == 384
    assert vectors["dense"].distance == qm.Distance.COSINE
    assert "sparse" in sparse_vectors

    payload_schema = info.payload_schema
    for field in ["date", "video_id", "channel"]:
        assert field in payload_schema, f"payload index missing: {field}"


def test_collection_metadata_stores_model_fingerprint(qdrant, collection_name):
    ensure_collection(qdrant, collection_name)

    fp = get_collection_fingerprint(qdrant, collection_name)
    assert fp["dense"] == model_fingerprint()
    assert fp["sparse"]["model"]


def test_ensure_collection_idempotent(qdrant, collection_name):
    ensure_collection(qdrant, collection_name)
    ensure_collection(qdrant, collection_name)  # no error, no wipe


def test_mismatched_fingerprint_raises_and_refuses_writes(qdrant, collection_name):
    ensure_collection(qdrant, collection_name)

    # Simulate a swapped model: fingerprint payload differs from current.
    altered = dict(model_fingerprint())
    altered["model"] = "some/other-model"
    altered["dim"] = 768
    _overwrite_meta_fingerprint(qdrant, collection_name, {"dense": altered, "sparse": {}})

    with pytest.raises(ModelMismatchError):
        ensure_collection(qdrant, collection_name)

    # Refuses writes: upsert path calls ensure_collection first.
    from mc.qdrant import upsert_video

    with pytest.raises(ModelMismatchError):
        upsert_video(
            qdrant,
            collection_name,
            video_id="abc",
            chunks=[_fake_chunk()],
            force=False,
        )


def test_fingerprint_included_via_dense_model_id(qdrant, collection_name):
    ensure_collection(qdrant, collection_name)
    fp = get_collection_fingerprint(qdrant, collection_name)
    assert fp["dense"]["model"] == DENSE_MODEL_ID


# --- helpers -------------------------------------------------------------

def get_collection_fingerprint(client: QdrantClient, name: str) -> dict:
    from mc.qdrant import read_meta_fingerprint

    return read_meta_fingerprint(client, name)


def _overwrite_meta_fingerprint(client: QdrantClient, name: str, payload: dict):
    """Test-only: rewrite the sentinel point to simulate a model swap."""
    client.upsert(
        collection_name=name,
        points=[
            qm.PointStruct(
                id=COLLECTION_META_POINT_ID,
                vector={"dense": [0.0] * DENSE_DIM},
                payload={"__meta__": True, "fingerprint": payload},
            )
        ],
    )


def _fake_chunk():
    from mc.ingest.models import Chunk

    return Chunk(
        video_id="abc123",
        text="texto de teste",
        url="https://www.youtube.com/watch?v=abc123&t=10s",
        start_seconds=10,
    )
