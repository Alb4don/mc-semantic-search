"""Story 5.3 — snapshot backup & restore (integration, local qdrant).

Runs against the local docker qdrant (docker compose up -d qdrant) at
http://localhost:6333. Cleanup discipline: every collection, alias AND
snapshot file created here is deleted again — snapshot files are
root-owned inside the ./snapshots volume, so they are removed through
the API (delete_snapshot), never by host-side rm.
"""

from __future__ import annotations

import os
import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from mc.backup import create_snapshot, restore, rotate
from mc.config import DENSE_DIM, SNAPSHOTS_DIR
from mc.ingest.models import Chunk
from mc.qdrant import COLLECTION_META_POINT_ID, ensure_collection, upsert_video

pytestmark = pytest.mark.integration


def _chunks(video_id: str, n: int) -> list[Chunk]:
    return [
        Chunk(
            video_id=video_id,
            text=f"backup drill chunk {i} sobre bitcoin e ethereum",
            url=f"https://www.youtube.com/watch?v={video_id}&t={i * 30}s",
            start_seconds=i * 30,
            date="2026-10-10",
            episode="4242",
            channel="Morning Crypto",
        )
        for i in range(n)
    ]


@pytest.fixture
def qdrant() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=60)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    return client


@pytest.fixture
def fake_dense(monkeypatch):
    """Fake dense embedder — real bm25 sparse still runs (Story 5.3 seeding)."""
    monkeypatch.setattr(
        "mc.qdrant.embed_passage", lambda texts: [[0.5] * DENSE_DIM for _ in texts]
    )


@pytest.fixture
def seeded(qdrant: QdrantClient, fake_dense) -> tuple[str, str]:
    """Fresh source collection (3 chunk points + meta point); fully torn down."""
    name = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    video_id = f"bk-{uuid.uuid4().hex[:6]}"
    ensure_collection(qdrant, name)
    upsert_video(qdrant, name, video_id, _chunks(video_id, 3))

    yield name, video_id

    # snapshot files are root-owned in the volume — delete through the API
    # while the collection still exists, then drop the collection itself.
    for desc in qdrant.list_snapshots(collection_name=name):
        qdrant.delete_snapshot(collection_name=name, snapshot_name=desc.name)
    qdrant.delete_collection(name)


def test_create_snapshot_places_file_in_snapshots_dir(qdrant, seeded):
    name, _ = seeded
    path = create_snapshot(client=qdrant, collection=name, keep=None)

    assert path.is_file(), f"snapshot file missing at {path}"
    assert path.suffix == ".snapshot"
    assert path.parent.name == name, "qdrant nests snapshots per collection"
    assert path.parent.parent.resolve() == SNAPSHOTS_DIR.resolve()
    listed = [s.name for s in qdrant.list_snapshots(collection_name=name)]
    assert path.name in listed


def test_restore_into_fresh_collection_is_searchable(qdrant, seeded):
    name, video_id = seeded
    snap = create_snapshot(client=qdrant, collection=name, keep=None)

    target = f"{name}-restored"
    alias = f"mc-drill-{uuid.uuid4().hex[:6]}"
    try:
        restore(snap, target, client=qdrant)

        scrolled, _ = qdrant.scroll(
            collection_name=target, limit=100, with_payload=True
        )
        chunk_points = [p for p in scrolled if p.id != COLLECTION_META_POINT_ID]
        assert len(chunk_points) == 3, "restored collection must hold the chunks"
        assert all(p.payload["video_id"] == video_id for p in chunk_points)

        # promotion path from the runbook: alias the restored collection
        # and confirm the alias resolves to it
        assert qdrant.update_collection_aliases(
            change_aliases_operations=[
                qm.CreateAliasOperation(
                    create_alias=qm.CreateAlias(
                        alias_name=alias, collection_name=target
                    )
                )
            ]
        )
        aliases = {
            a.alias_name: a.collection_name for a in qdrant.get_aliases().aliases
        }
        assert aliases[alias] == target
    finally:
        qdrant.update_collection_aliases(
            change_aliases_operations=[
                qm.DeleteAliasOperation(delete_alias=qm.DeleteAlias(alias_name=alias))
            ]
        )
        qdrant.delete_collection(target)


def test_restore_refuses_existing_target(qdrant, seeded):
    """The documented contract is enforced: an existing target is an error,
    never a silent rollback of the live collection."""
    name, _ = seeded
    snap = create_snapshot(client=qdrant, collection=name, keep=None)

    with pytest.raises(ValueError, match="already exists"):
        restore(snap, name, client=qdrant)

    # the refused restore must have left the live collection intact
    count = qdrant.count(collection_name=name, exact=True)
    assert count.count == 4  # 3 chunk points + meta point


def test_rotation_keeps_last_seven(qdrant, seeded, tmp_path):
    for i in range(9):
        f = tmp_path / f"snap-{i:02d}.snapshot"
        f.write_bytes(b"x")
        os.utime(f, (1_000_000 + i, 1_000_000 + i))

    removed = rotate(tmp_path, keep=7)

    left = sorted(p.name for p in tmp_path.glob("*.snapshot"))
    assert left == [f"snap-{i:02d}.snapshot" for i in range(2, 9)], (
        "exactly the 7 newest snapshots must survive"
    )
    assert {p.name for p in removed} == {"snap-00.snapshot", "snap-01.snapshot"}
    assert not list(tmp_path.glob("*.checksum"))


def test_keep_param_honored_via_api_rotation(qdrant, seeded):
    name, _ = seeded
    create_snapshot(client=qdrant, collection=name, keep=1)
    create_snapshot(client=qdrant, collection=name, keep=1)

    listed = qdrant.list_snapshots(collection_name=name)
    assert len(listed) == 1, "keep=1 must leave exactly the newest snapshot"
