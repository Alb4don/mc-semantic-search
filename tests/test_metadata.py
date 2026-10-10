"""Story 2.2 — yt-dlp metadata enrichment with a resumable on-disk cache."""

from __future__ import annotations

import json
from pathlib import Path

from mc.ingest.metadata import enrich


def _payload(video_id: str, **overrides) -> dict:
    raw = {
        "upload_date": "20260109",
        "title": f"BITCOIN: O PLANO SECRETO #{video_id}",
        "uploader": "Morning Crypto",
        "channel": "Morning Crypto Channel",
    }
    raw.update(overrides)
    return raw


def test_fetch_invoked_once_per_unique_video(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        return _payload(video_id)

    result = enrich(["a", "b", "a"], cache_dir=tmp_path, fetch=fetch)

    assert calls == ["a", "b"]
    assert set(result) == {"a", "b"}


def test_result_cached_second_call_skips_fetch(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        return _payload(video_id)

    first = enrich(["a", "b"], cache_dir=tmp_path, fetch=fetch)
    calls.clear()

    second = enrich(["a", "b"], cache_dir=tmp_path, fetch=fetch)

    assert calls == []
    assert second == first


def test_upload_date_normalized_and_channel(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return {
            "upload_date": "20260109",
            "title": "SOME TITLE #123",
            "uploader": "Morning Crypto",
            "channel": "Channel Field",
        }

    meta = enrich(["abc"], cache_dir=tmp_path, fetch=fetch)["abc"]

    assert meta["date"] == "2026-01-09"
    assert meta["channel"] == "Morning Crypto"
    assert (tmp_path / "abc.json").exists()
    assert json.loads((tmp_path / "abc.json").read_text()) == meta


def test_episode_parsed_from_title_digits(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return _payload(video_id, title="BITCOIN: O PLANO SECRETO #1234")

    meta = enrich(["x1"], cache_dir=tmp_path, fetch=fetch)["x1"]
    assert meta["episode"] == "1234"


def test_episode_none_when_title_has_no_digits(tmp_path: Path):
    def fetch(video_id: str) -> dict:
        return _payload("x2", title="BITCOIN: O PLANO SECRETO")

    meta = enrich(["x2"], cache_dir=tmp_path, fetch=fetch)["x2"]
    assert meta["episode"] is None


def test_failure_retries_three_times_then_missing(tmp_path: Path):
    calls: list[str] = []

    def fetch(video_id: str) -> dict:
        calls.append(video_id)
        if video_id == "bad":
            raise RuntimeError("network down")
        return _payload(video_id)

    result = enrich(
        ["bad", "good"], cache_dir=tmp_path, fetch=fetch, backoff=(0, 0, 0)
    )

    assert calls.count("bad") == 3
    assert "good" in calls
    assert set(result) == {"good"}
    assert (tmp_path / "missing.txt").read_text().splitlines() == ["bad"]
