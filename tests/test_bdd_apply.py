"""BDD step definitions for tests/bdd/apply.feature (Story 3.7).

Runs against the local qdrant container (docker compose up -d qdrant).
Same embedder-faking pattern as the backfill BDD suite: `mc.qdrant.embed_passage`
is patched (upsert_video resolves it in that module namespace) with a fake
returning cheap 384-dim vectors, and a SPY records every embedded text so we
can assert the second `mc apply` run embeds nothing (applied-once tracking).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

scenarios("bdd/apply.feature")

pytestmark = pytest.mark.integration


def _client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    client.get_collections()
    return client


def _record(video_id: str, i: int) -> dict:
    return {
        "video_id": video_id,
        "text": f"chunk {i} do vídeo {video_id}",
        "url": f"https://youtube.com/watch?v={video_id}&t={i * 30}",
        "start_seconds": i * 30,
        "date": "2026-10-09",
        "episode": "ep",
        "channel": "Morning Crypto",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
        encoding="utf-8",
    )


@pytest.fixture
def ctx(tmp_path):
    try:
        client = _client()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    ctx = {
        "client": client,
        "collection": f"mc-bdd-{uuid.uuid4().hex[:8]}",
        "state_db": tmp_path / "ingest.db",
        "queue_dir": queue_dir,
        "lock_path": tmp_path / "update.lock",
        "embedded_texts": [],
        "reports": [],
    }
    yield ctx
    client.delete_collection(ctx["collection"])
    client.close()


def _run_apply(ctx) -> dict:
    """One `mc apply` run with the dense embedder faked (spied)."""
    import mc.qdrant as q
    from mc.ingest.apply import run_apply

    def fake_embed_passage(texts):
        ctx["embedded_texts"].extend(texts)
        return [[0.5] * 384 for _ in texts]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", fake_embed_passage)
    try:
        report = run_apply(
            queue_dir=ctx["queue_dir"],
            state=ctx["state_db"],
            client=ctx["client"],
            collection=ctx["collection"],
            lock_path=ctx["lock_path"],
        )
    finally:
        monkeypatch.undo()
    ctx["reports"].append(report)
    return report


def _point_count(ctx) -> int:
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


# --- Scenario: Increment applied exactly once --------------------------------


@given(parsers.parse("a queued increment with {n:d} chunks"))
def queued_increment(n, ctx):
    records = [_record("bddvid00001", i) for i in range(n)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-09.jsonl", records)
    ctx["records"] = records


@when('"mc apply" runs twice')
def runs_twice(ctx):
    _run_apply(ctx)
    _run_apply(ctx)


@then("the second run is a no-op (chunks tracked by content hash)")
def second_run_is_noop(ctx):
    first, second = ctx["reports"]
    assert first["applied_chunks"] == len(ctx["records"])
    assert second["applied_chunks"] == 0
    # embedding spy: exactly the 3 chunks, ZERO new texts on run 2
    assert len(ctx["embedded_texts"]) == len(ctx["records"])
    assert "nothing to apply" in second["notes"]


@then(parsers.parse("all {n:d} chunks are searchable"))
def chunks_searchable(n, ctx):
    assert _point_count(ctx) == n


# --- Scenario: Corrupt increment is quarantined -------------------------------


@given("a malformed JSONL file in the queue directory")
def malformed_file(ctx):
    broken = ctx["queue_dir"] / "broken.jsonl"
    broken.write_text('{"video_id": "x"\n{not valid json\n', encoding="utf-8")
    ctx["broken"] = broken


@given(parsers.parse("another queued file with {n:d} valid chunks"))
def another_queued_file(n, ctx):
    records = [_record("bddvid00002", i) for i in range(n)]
    _write_jsonl(ctx["queue_dir"] / "2026-10-10.jsonl", records)
    ctx["other_records"] = records


@when('"mc apply" runs')
def apply_runs(ctx):
    _run_apply(ctx)


@then("the file is moved to queue/failed/ with the error logged")
def corrupt_file_quarantined(ctx, caplog):
    failed = ctx["queue_dir"] / "failed" / "broken.jsonl"
    assert failed.exists()
    assert not ctx["broken"].exists()
    assert ctx["reports"][-1]["failed_files"] == 1
    assert any("broken.jsonl" in r.getMessage() for r in caplog.records)


@then("other queued files are still applied")
def other_files_applied(ctx):
    report = ctx["reports"][-1]
    assert report["applied_chunks"] == len(ctx["other_records"])
    assert report["failed_files"] == 1
    assert _point_count(ctx) == len(ctx["other_records"])
    # the valid file was applied, not quarantined
    assert (ctx["queue_dir"] / "2026-10-10.jsonl").exists()
