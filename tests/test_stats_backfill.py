"""Story 6.4 — `mc stats-backfill`: bootstrap archive stats for the videos
already indexed, without any yt-dlp fetch by default.

Duration precedence: metadata cache > local transcripts cache
(`max(start + duration)` over the JSON segment list) > (with --fetch-missing)
one yt-dlp fetch that also upgrades the cache file. Resumable: a video whose
stats row already has a duration is skipped, and a fetched cache entry is
never re-fetched.
"""

from __future__ import annotations

import json
from pathlib import Path

from mc.ingest.stats_backfill import run_stats_backfill
from mc.state import StateStore


def _write_cache(cache_dir: Path, video_id: str, meta: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{video_id}.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _write_transcript(transcripts_dir: Path, video_id: str, segments: list[dict]) -> None:
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (transcripts_dir / f"{video_id}.srt").write_text(
        json.dumps(segments), encoding="utf-8"
    )


def _ctx(tmp_path: Path):
    state = StateStore(tmp_path / "ingest.db")
    return state, tmp_path / "meta", tmp_path / "transcripts"


def test_universe_is_processed_union_cache_ids(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.mark_processed("p1", "update")
    state.mark_processed("p2", "update")
    _write_cache(cache_dir, "p2", {"episode": "1", "date": "2026-01-01"})
    _write_cache(cache_dir, "c1", {"episode": "2", "date": "2026-01-02"})

    report = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )

    assert report["videos_considered"] == 3  # p1 ∪ p2 ∪ c1 (deduped)


def test_duration_derived_from_transcript_segments(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})
    _write_transcript(
        transcripts_dir,
        "v1",
        [
            {"text": "a", "start": 0.0, "duration": 10.0},
            {"text": "b", "start": 10.0, "duration": 5.0},
        ],
    )

    report = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    stats = state.archive_stats()
    assert stats["seconds"] == 15  # int(max(0+10, 10+5))
    assert stats["last_episode"] == 971
    assert report["videos_backfilled"] == 1


def test_cache_duration_precedes_transcript(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01", "duration": 999})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 10.0}])

    run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    assert state.archive_stats()["seconds"] == 999


def test_transcript_precedes_fetch(tmp_path):
    """A local transcript beats a fetch even when --fetch-missing is set."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01"})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 42.0}])

    def fake_fetch(_vid):
        raise AssertionError("must not fetch when a transcript exists")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert state.archive_stats()["seconds"] == 42
    assert report["fetched"] == 0


def test_limit_bounds_a_run(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    for i in range(3):
        _write_cache(cache_dir, f"c{i}", {"episode": str(i), "date": "2026-01-01"})

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        limit=2,
    )
    assert report["videos_considered"] == 2
    assert state.archive_stats()["videos"] == 2


def test_second_run_is_resumable(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 15.0}])

    first = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    assert first["videos_backfilled"] == 1
    assert state.archive_stats()["seconds"] == 15

    second = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    assert second["videos_backfilled"] == 0
    assert second["videos_skipped"] == 1
    assert state.archive_stats()["seconds"] == 15  # never doubled


def test_fetch_missing_upgrades_cache_with_duration(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    calls: list[str] = []

    def fake_fetch(video_id: str) -> dict:
        calls.append(video_id)
        return {"upload_date": "20261020", "title": "EP #971", "duration": 5400}

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert report["fetched"] == 1
    assert state.archive_stats()["seconds"] == 5400
    # the cache file is upgraded with duration, so the fetch happens once ever
    cached = json.loads((cache_dir / "v1.json").read_text())
    assert cached["duration"] == 5400

    calls.clear()
    run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=fake_fetch,
        sleeper=lambda _: None,
    )
    assert calls == []  # cache now has duration → no second fetch


def test_missing_transcript_without_flag_is_null_no_exception(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    report = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    stats = state.archive_stats()
    assert stats["videos"] == 1
    assert stats["seconds"] is None  # no duration source, no flag → NULL
    assert stats["last_episode"] == 971  # episode/date still recorded
    assert report["still_missing"] == 1


def test_garbage_transcript_is_skipped_not_fatal(tmp_path):
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(cache_dir, "v1", {"episode": "1", "date": "2026-01-01"})
    (transcripts_dir).mkdir(parents=True, exist_ok=True)
    (transcripts_dir / "v1.srt").write_text("not json at all", encoding="utf-8")

    report = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    assert state.archive_stats()["seconds"] is None
    assert report["still_missing"] == 1


def test_fetch_missing_skips_already_complete_videos(tmp_path):
    """--fetch-missing targets videos *still lacking* a duration: a video whose
    stats row already carries one is skipped in both modes, so a re-run
    converges and never touches YouTube again."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    state.record_video_stats("v1", duration=5400, episode=971, date="2026-10-20")
    _write_cache(cache_dir, "v1", {"episode": "971", "date": "2026-10-20"})  # no duration

    def boom(_vid):
        raise AssertionError("must not fetch a video that already has a duration")

    report = run_stats_backfill(
        state=state,
        cache_dir=cache_dir,
        transcripts_dir=transcripts_dir,
        fetch_missing=True,
        fetch=boom,
        sleeper=lambda _: None,
    )
    assert report["fetched"] == 0
    assert report["videos_skipped"] == 1
    assert report["videos_backfilled"] == 0
    assert state.archive_stats()["seconds"] == 5400


def test_garbage_cache_duration_falls_through_to_transcript(tmp_path):
    """A junk duration in a (hand-edited) cache entry must not crash — it
    falls through to the transcript derivation instead."""
    state, cache_dir, transcripts_dir = _ctx(tmp_path)
    _write_cache(
        cache_dir,
        "v1",
        {"episode": "1", "date": "2026-01-01", "duration": "not-a-number"},
    )
    _write_transcript(transcripts_dir, "v1", [{"start": 0.0, "duration": 15.0}])

    report = run_stats_backfill(
        state=state, cache_dir=cache_dir, transcripts_dir=transcripts_dir
    )
    assert state.archive_stats()["seconds"] == 15
    assert report["videos_backfilled"] == 1
