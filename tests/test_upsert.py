"""Story 1.4 — per-video atomic upserts (NFR-3)."""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

from mc.ingest.models import Chunk
from mc.qdrant import ModelMismatchError, ensure_collection, upsert_video

pytestmark = pytest.mark.integration


def _chunks(video_id: str, n: int, *, date="2026-10-01") -> list[Chunk]:
    return [
        Chunk(
            video_id=video_id,
            text=f"chunk número {i} do vídeo {video_id} sobre bitcoin",
            url=f"https://www.youtube.com/watch?v={video_id}&t={i * 30}s",
            start_seconds=i * 30,
            date=date,
            episode="1234",
            channel="Morning Crypto",
        )
        for i in range(n)
    ]


def _video_point_count(client: QdrantClient, name: str, video_id: str) -> int:
    from qdrant_client.http import models as qm

    count = 0
    offset = None
    while True:
        scrolled, offset = client.scroll(
            collection_name=name,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="video_id", match=qm.MatchValue(value=video_id)
                    )
                ]
            ),
            limit=1000,
            offset=offset,
        )
        count += len(scrolled)
        if offset is None:
            return count


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    return client


@pytest.fixture
def coll(qdrant: QdrantClient) -> str:
    name = f"mc-test-{uuid.uuid4().hex[:8]}"
    ensure_collection(qdrant, name)
    yield name
    qdrant.delete_collection(name)


def test_reupsert_replaces_all_points(qdrant, coll):
    vid = "reupsert1"
    upsert_video(qdrant, coll, vid, _chunks(vid, 5))
    assert _video_point_count(qdrant, coll, vid) == 5

    upsert_video(qdrant, coll, vid, _chunks(vid, 3))
    assert _video_point_count(qdrant, coll, vid) == 3, (
        "stale points must be deleted in the same operation"
    )


def test_upsert_payload_complete(qdrant, coll):
    vid = "payload1"
    upsert_video(qdrant, coll, vid, _chunks(vid, 2))
    (points, _) = qdrant.scroll(
        collection_name=coll,
        scroll_filter={"must": [{"key": "video_id", "match": {"value": vid}}]},
        limit=10,
        with_payload=True,
    )
    p = points[0].payload
    for field in ["text", "url", "start_seconds", "date", "episode", "channel"]:
        assert field in p and p[field] is not None, f"payload missing {field}"
    assert p["url"].startswith("https://www.youtube.com/watch?v=")


def test_embed_failure_leaves_video_absent(qdrant, coll, monkeypatch):
    vid = "boom1"
    upsert_video(qdrant, coll, vid, _chunks(vid, 2))

    def explode(texts):
        raise RuntimeError("embedding worker died")

    monkeypatch.setattr("mc.qdrant.embed_passage", explode)
    with pytest.raises(RuntimeError):
        upsert_video(qdrant, coll, vid, _chunks(vid, 4))

    # Either absent or complete — never half (here: old complete set remains).
    assert _video_point_count(qdrant, coll, vid) == 2


def test_model_mismatch_refuses_upsert(qdrant, coll, monkeypatch):
    from mc.qdrant import read_meta_fingerprint

    vid = "mismatch1"
    stored = read_meta_fingerprint(qdrant, coll)
    tampered = {"dense": dict(stored["dense"], dim=999), "sparse": {}}

    from qdrant_client.http import models as qm

    qdrant.upsert(
        collection_name=coll,
        points=[
            qm.PointStruct(
                id=0,
                vector={"dense": [0.0] * 384},
                payload={"__meta__": True, "fingerprint": tampered},
            )
        ],
    )
    with pytest.raises(ModelMismatchError):
        upsert_video(qdrant, coll, vid, _chunks(vid, 1))
    assert _video_point_count(qdrant, coll, vid) == 0


def test_processed_marker_is_caller_not_upsert():
    """The state marker must be written by the orchestrator after a successful
    commit — upsert_video itself never touches the state store (Story 1.4)."""
    import inspect

    import mc.qdrant as q

    source = inspect.getsource(q)
    assert "mark_processed" not in source
    assert "StateStore" not in source
