"""Story 3.8 — `mc reindex`: model-swap migration with atomic alias switch.

Integration tests against a local qdrant container: each test seeds a
fresh source collection with a FAKE dense embedder (monkeypatched on the
mc.qdrant module where upsert_video resolves it) plus the real cached
bm25 sparse encoder, then checks copy/refuse/alias semantics.
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from mc.ingest.models import Chunk
from mc.ingest.reindex import ReindexError, run_reindex
from mc.qdrant import (
    COLLECTION_META_POINT_ID,
    ensure_collection,
    read_meta_fingerprint,
    upsert_video,
)

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


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    return client


@pytest.fixture
def source(qdrant: QdrantClient) -> str:
    name = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    yield name
    qdrant.delete_collection(name)
    qdrant.delete_collection(f"{name}-v2")


def _fake_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mc.qdrant.embed_passage", lambda texts: [[0.5] * 384 for _ in texts]
    )


def _seed_source(qdrant: QdrantClient, name: str, video_ids: list[str]) -> None:
    for vid in video_ids:
        upsert_video(qdrant, name, vid, _chunks(vid, 3))


def _all_points(client: QdrantClient, name: str) -> list:
    points: list = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            return points


def _chunk_points(client: QdrantClient, name: str) -> list:
    """All points EXCLUDING the meta sentinel (id == 0)."""
    return [p for p in _all_points(client, name) if p.id != COLLECTION_META_POINT_ID]


def test_copies_points_and_writes_fingerprint(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    vids = ["video000001", "video000002"]
    _seed_source(qdrant, source, vids)

    report = run_reindex(client=qdrant, source_collection=source)

    target = f"{source}-v2"
    assert report["target"] == target
    assert report["videos"] == 2
    assert report["points_copied"] == 6

    # Target carries a readable fingerprint meta point for the current models.
    fingerprint = read_meta_fingerprint(qdrant, target)
    assert fingerprint["dense"]["model"]

    # Payloads round-trip per chunk (keyed by video_id + start_seconds).
    payloads = {
        (p.payload["video_id"], p.payload["start_seconds"]): p.payload
        for p in _chunk_points(qdrant, target)
    }
    assert set(payloads) == {(v, i * 30) for v in vids for i in range(3)}
    payload = payloads[("video000001", 0)]
    assert payload["text"] == "chunk número 0 do vídeo video000001 sobre bitcoin"
    assert payload["url"].endswith("v=video000001&t=0s")
    assert payload["start_seconds"] == 0
    assert payload["date"] == "2026-10-01"
    assert payload["episode"] == "1234"
    assert payload["channel"] == "Morning Crypto"


def test_refuses_when_target_has_different_fingerprint(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    _seed_source(qdrant, source, ["video000001"])
    target = f"{source}-v2"
    ensure_collection(qdrant, target)
    decoy = {
        "dense": {"model": "bogus/other", "dim": 384, "hash": "decoy"},
        "sparse": {"model": "bogus/bm25", "language": "x", "hash": "decoy2"},
    }
    qdrant.upsert(
        collection_name=target,
        points=[
            qm.PointStruct(
                id=COLLECTION_META_POINT_ID,
                vector={"dense": [0.0] * 384},
                payload={"__meta__": True, "fingerprint": decoy},
            )
        ],
    )

    with pytest.raises(ReindexError, match="different fingerprint"):
        run_reindex(client=qdrant, source_collection=source)

    # Refusal happened before any copy: target untouched, no alias created.
    aliases = qdrant.get_aliases().aliases
    assert all(a.collection_name != target for a in aliases)


def test_alias_points_at_target_after_swap(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    _seed_source(qdrant, source, ["video000001"])

    report = run_reindex(client=qdrant, source_collection=source)
    target = f"{source}-v2"
    alias = report["alias"]
    assert alias == f"{source}-alias"

    pairs = {(a.alias_name, a.collection_name) for a in qdrant.get_aliases().aliases}
    assert (alias, target) in pairs
    assert (alias, source) not in pairs


def test_existing_alias_is_swapped_atomically(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    _seed_source(qdrant, source, ["video000001"])
    alias = f"{source}-alias"
    qdrant.update_collection_aliases(
        change_aliases_operations=[
            qm.CreateAliasOperation(
                create_alias=qm.CreateAlias(collection_name=source, alias_name=alias)
            )
        ]
    )

    report = run_reindex(client=qdrant, source_collection=source)

    assert report["alias"] == alias
    pairs = {(a.alias_name, a.collection_name) for a in qdrant.get_aliases().aliases}
    assert (alias, f"{source}-v2") in pairs
    assert (alias, source) not in pairs


def test_source_collection_retained_after_swap(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    vids = ["video000001", "video000002"]
    _seed_source(qdrant, source, vids)

    run_reindex(client=qdrant, source_collection=source)

    # Old collection is never deleted (rollback path).
    info = qdrant.get_collection(source)
    assert info.status is not None
    assert len(_chunk_points(qdrant, source)) == 6


def test_meta_sentinel_excluded_from_copy(qdrant, source, monkeypatch):
    _fake_dense(monkeypatch)
    _seed_source(qdrant, source, ["video000001", "video000002"])

    report = run_reindex(client=qdrant, source_collection=source)

    target = f"{source}-v2"
    all_target_points = _all_points(qdrant, target)
    # exactly the chunk points + its own meta point
    assert len(all_target_points) == 7
    assert sum(1 for p in all_target_points if p.id == COLLECTION_META_POINT_ID) == 1
    assert report["points_copied"] == 6
    assert report["videos"] == 2
