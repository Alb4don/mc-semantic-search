"""BDD step definitions for tests/bdd/update.feature (Story 3.4).

Runs against the local qdrant container. Discovery, download, metadata
and the embedder are fakes/spies; the assertions prove per-video
transactions, processed markers, failure isolation and the run summary.
"""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.ingest import update
from mc.state import StateStore

scenarios("bdd/update.feature")

pytestmark = pytest.mark.integration


def _client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=30)
    client.get_collections()
    return client


def _segments(n_chunks: int, seed: str) -> list[dict]:
    """Enough segments for n_chunks (30 per chunk), each text unique."""
    return [
        {"text": f"{seed} segment {i}", "start": float(i), "duration": 2.0}
        for i in range(n_chunks * 30)
    ]


@pytest.fixture
def ctx(tmp_path):
    try:
        client = _client()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx = {
        "client": client,
        "collection": collection,
        "state": StateStore(tmp_path / "ingest.db"),
        "tmp_path": tmp_path,
        "downloaded": [],
        "transcripts_dir": tmp_path / "transcripts",
        "queue_dir": tmp_path / "queue",
        "ignore_file": tmp_path / "transcripts" / "ignored_videos.txt",
        "exit_code": None,
    }
    yield ctx
    client.delete_collection(collection)


def _make_updater(ctx, transcript_plan: dict[str, list[dict] | None]):
    """Fake downloader recording calls; None return simulates failure."""
    def downloader(video_id: str):
        ctx["downloaded"].append(video_id)
        return transcript_plan.get(video_id)

    return downloader


def _run(ctx, metadata_fetch=None):
    def fake_embed_passage(texts):
        return [[0.5] * 384 for _ in texts]

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", fake_embed_passage)
    try:
        ctx["exit_code"] = update.run_update(
            channels=("https://www.youtube.com/@test",),
            state=ctx["state"],
            client=ctx["client"],
            collection=ctx["collection"],
            discover=lambda channels: ctx["discovered"],
            downloader=ctx["downloader"],
            metadata_fetch=metadata_fetch or (lambda ids: {}),
            lock_path=ctx["tmp_path"] / "update.lock",
            transcripts_dir=ctx["transcripts_dir"],
            queue_dir=ctx["queue_dir"],
        )
    finally:
        monkeypatch.undo()


@given(
    parsers.parse('channels return video IDs ["{video_id}"] and none processed')
)
def channels_one_new(ctx, video_id):
    ctx["discovered"] = [video_id]
    ctx["downloader"] = _make_updater(ctx, {video_id: _segments(3, video_id)})


@when("`mc update` runs successfully")
def run_success(ctx):
    _run(ctx)


@then(
    parsers.parse(
        '"{video_id}" is searchable in Qdrant with {expected:d} chunk points'
    )
)
def searchable_with_points(ctx, video_id, expected):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"], limit=1000
    )
    chunks = [p for p in points if p.id != 0]  # exclude meta sentinel
    assert len(chunks) == expected
    assert all(p.payload["video_id"] == video_id for p in chunks)


@then(
    parsers.parse(
        'the state store marks "{video_id}" processed after the upsert commits'
    )
)
def state_marks_processed(ctx, video_id):
    assert ctx["state"].is_processed(video_id)


@then(
    parsers.parse(
        "the queue staging file contains the {expected:d} chunk records"
    )
)
def queue_staging_records(ctx, expected):
    pending = ctx["queue_dir"] / "pending.jsonl"
    assert pending.exists()
    lines = [
        json.loads(line)
        for line in pending.read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == expected
    assert set(lines[0]) >= {
        "video_id",
        "text",
        "url",
        "start_seconds",
        "date",
        "episode",
        "channel",
    }


@given(
    parsers.parse(
        'channels return ["{first}", "{second}"] and both are marked processed'
    )
)
def channels_all_processed(ctx, first, second):
    ctx["discovered"] = [first, second]
    for vid in (first, second):
        ctx["state"].mark_processed(vid, "update")
    ctx["downloader"] = _make_updater(ctx, {})


@when("`mc update` runs")
def run_plain(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        _run(ctx)


@then(
    parsers.parse(
        'no transcript download happens for "{first}" or "{second}"'
    )
)
def no_downloads(ctx, first, second):
    assert ctx["downloaded"] == []


@then(parsers.parse('the run log reports "{expected}"'))
def run_log_reports(ctx, caplog, expected):
    assert expected in caplog.text


@given(
    parsers.parse(
        'video "{bad}" fails after retries and "{good}" succeeds'
    )
)
def one_bad_one_good(ctx, bad, good):
    ctx["discovered"] = [bad, good]
    ctx["downloader"] = _make_updater(
        ctx, {bad: None, good: _segments(3, good)}
    )


@when("`mc update` finishes")
def run_with_failure(ctx, caplog):
    with caplog.at_level(logging.INFO, logger="mc.ingest.update"):
        _run(ctx)


@then(
    parsers.parse('"{good}" is indexed, "{bad}" is in the ignore list')
)
def good_indexed_bad_ignored(ctx, good, bad):
    points, _ = ctx["client"].scroll(
        collection_name=ctx["collection"], limit=1000
    )
    chunks = [p for p in points if p.id != 0]
    assert chunks and all(p.payload["video_id"] == good for p in chunks)
    assert ctx["ignore_file"].exists()
    assert bad in ctx["ignore_file"].read_text()


@then("the run exits non-zero with a summary of the failure")
def exits_non_zero(ctx, caplog):
    assert ctx["exit_code"] == 1
    assert "bad1" in caplog.text and "failed" in caplog.text.lower()


def test_auth_failure_is_not_ignore_listed(tmp_path, caplog):
    """Story 5.6: stale-cookie (auth) failures get a distinct reason and the
    video is never added to the ignore list — the episode is retried after
    cookie rotation instead of skipped forever."""
    import logging as _logging
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")
    queue_dir = tmp_path / "queue"

    def stale_cookies(video_id: str):
        raise RuntimeError("Sign in to confirm you're not a bot")

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        with caplog.at_level(_logging.ERROR, logger="mc.ingest.update"):
            code = update.run_update(
                channels=("https://www.youtube.com/@test",),
                state=state,
                client=client,
                collection=collection,
                discover=lambda channels: ["stale1"],
                downloader=stale_cookies,
                metadata_fetch=lambda ids: {},
                lock_path=tmp_path / "update.lock",
                transcripts_dir=tmp_path / "transcripts",
                queue_dir=queue_dir,
            )
    finally:
        monkeypatch.undo()

    assert code == 1
    assert not (tmp_path / "transcripts" / "ignored_videos.txt").exists()
    assert "cookies likely stale" in caplog.text
    assert not state.is_processed("stale1")
    client.delete_collection(collection)


def test_no_subtitles_failure_is_ignore_listed(tmp_path):
    """Non-auth download failures keep the v3 ignore-list semantics."""
    import uuid as _uuid

    from qdrant_client import QdrantClient as QC

    client = QC(url="http://localhost:6333", timeout=30)
    collection = f"mc-bdd-{_uuid.uuid4().hex[:8]}"
    state = StateStore(tmp_path / "ingest.db")

    import mc.qdrant as q

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(q, "embed_passage", lambda texts: [[0.5] * 384 for _ in texts])
    try:
        code = update.run_update(
            channels=("https://www.youtube.com/@test",),
            state=state,
            client=client,
            collection=collection,
            discover=lambda channels: ["quiet1"],
            downloader=lambda vid: None,
            metadata_fetch=lambda ids: {},
            lock_path=tmp_path / "update.lock",
            transcripts_dir=tmp_path / "transcripts",
            queue_dir=tmp_path / "queue",
        )
    finally:
        monkeypatch.undo()

    assert code == 1
    assert "quiet1" in (tmp_path / "transcripts" / "ignored_videos.txt").read_text()
    client.delete_collection(collection)
