"""BDD step definitions for tests/bdd/backfill.feature (Story 2.4).

Runs against the local qdrant container (docker compose up -d qdrant).
The embedder is faked with cheap deterministic vectors; the SPY records
every successfully embedded text so we can assert no chunk is ever
embedded twice across a crash+resume.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.ingest import backfill
from mc.state import StateStore

scenarios("bdd/backfill.feature")

pytestmark = pytest.mark.integration


def _client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    client.get_collections()
    return client


def _make_legacy_db(tmp_path: Path, video_sizes: list[int]) -> Path:
    """Build a mini legacy DB with the same FTS5 schema as the real one."""
    import sqlite3

    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE VIRTUAL TABLE transcripts USING fts5(title, url, description)"
    )
    n = 0
    for v_idx, size in enumerate(video_sizes):
        # 11-char ids: the legacy reader validates YouTube id format
        vid = f"video{v_idx:06d}"
        for c in range(size):
            conn.execute(
                "INSERT INTO transcripts(title, url, description) VALUES (?, ?, ?)",
                (
                    f"video {v_idx}",
                    f"https://www.youtube.com/watch?v={vid}&t={c * 30}",
                    f"texto do chunk {n} do vídeo {vid}",
                ),
            )
            n += 1
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def ctx(tmp_path):
    try:
        client = _client()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    state_db = tmp_path / "ingest.db"
    ctx = {
        "client": client,
        "collection": collection,
        "state_db": state_db,
        "embedded_texts": [],
        "kill_after_commit": None,
    }
    yield ctx
    client.delete_collection(collection)


@given(
    parsers.parse(
        "a legacy DB with 12 chunks across 2 videos and a batch size of 5"
    )
)
def legacy_db_12_chunks_2_videos(tmp_path, ctx):
    ctx["db"] = _make_legacy_db(tmp_path, [7, 5])
    ctx["batch_size"] = 5


@given(parsers.parse("the legacy DB with 12 chunks in 1 video is fully imported"))
def legacy_db_12_chunks_1_video(tmp_path, ctx):
    ctx["db"] = _make_legacy_db(tmp_path, [12])
    ctx["batch_size"] = 5


def _run(ctx):
    def fake_embed_passage(texts):
        ctx["embedded_texts"].extend(texts)
        return [[0.5] * 384 for _ in texts]

    import mc.qdrant as q

    state = StateStore(ctx["state_db"])

    def kill_after_commit(video_id: str) -> None:
        if ctx.get("kill_after_commit") == video_id:
            raise backfill.BackfillInterrupted(f"killed after {video_id}")

    import pytest

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", fake_embed_passage)
    try:
        report = backfill.run_backfill(
            from_db=ctx["db"],
            batch_size=ctx["batch_size"],
            client=ctx["client"],
            collection=ctx["collection"],
            state=state,
            kill_after_commit=kill_after_commit,
        )
    finally:
        monkeypatch.undo()
    ctx["report"] = report
    return report


@when("backfill processes 7 chunks and is killed")
def run_until_killed(ctx):
    ctx["kill_after_commit"] = "video000000"  # 7 chunks committed
    try:
        _run(ctx)
    except backfill.BackfillInterrupted:
        pass
    else:
        raise AssertionError("expected interruption")


@when("backfill is started again")
def run_again(ctx):
    ctx["kill_after_commit"] = None
    _run(ctx)


@when("backfill finishes")
def run_to_completion(ctx):
    _run(ctx)


@then("the remaining 5 chunks are embedded and indexed")
def remaining_5_embedded(ctx):
    assert ctx["report"]["indexed_chunks"] == 5
    # 7 points persisted before the kill + the 5 resumed now
    count = _point_count(ctx)
    assert count == 12


def _point_count(ctx):
    client = ctx["client"]
    count = 0
    offset = None
    while True:
        scrolled, offset = client.scroll(
            collection_name=ctx["collection"], limit=1000, offset=offset
        )
        count += sum(1 for p in scrolled if p.id != 0)  # exclude meta sentinel
        if offset is None:
            return count


@then("no chunk is embedded twice (total embedding calls: 12)")
def no_double_embedding(ctx):
    assert len(ctx["embedded_texts"]) == 12, (
        f"embedded {len(ctx['embedded_texts'])} texts: expected exactly 12"
    )
    assert len(set(ctx["embedded_texts"])) == 12


@then("the Qdrant collection contains 12 points")
def collection_has_12_points(ctx):
    assert _point_count(ctx) == 12


@then("the state store reports 1 video processed")
def one_video_processed(ctx):
    state = StateStore(ctx["state_db"])
    assert state.count_processed() == 1


@then("a final report prints indexed/quarantine/skipped counts")
def report_has_counts(ctx):
    report = ctx["report"]
    assert set(report) >= {"indexed_chunks", "quarantined", "skipped_videos"}
    assert report["indexed_chunks"] == 12
    assert report["skipped_videos"] == 0
