"""Story 3.1 — channel video listing via yt-dlp --flat-playlist."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc.ingest.discover import DiscoveryError, list_channel_videos

FIXTURE = Path(__file__).parent / "fixtures" / "flat_playlist.json"


@pytest.fixture
def playlist_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def _fake_run(returncode=0, stdout="", stderr=""):
    def run(cmd, **kwargs):
        import types

        proc = types.SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr=stderr
        )
        return proc

    return run


def test_lists_video_ids_from_flat_playlist(playlist_fixture, monkeypatch):
    stdout = json.dumps(playlist_fixture)
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, stdout, ""))
    ids = list_channel_videos(["https://www.youtube.com/@canal"])
    assert ids == {"vid1", "vid2", "vid3"}


def test_multiple_channels_union(playlist_fixture, monkeypatch):
    stdout = json.dumps(playlist_fixture)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_run(0, stdout, "")(cmd, **kwargs)

    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", run)
    ids = list_channel_videos(["https://www.youtube.com/@a", "https://youtube.com/@b"])
    assert len(ids) == 3
    assert len(calls) == 2


def test_channels_come_from_config_not_hardcoded(monkeypatch, tmp_path):
    """Channels must be configurable (env/config.yaml), never baked in code."""
    import mc.config as config

    monkeypatch.setenv("MC_CHANNELS", "https://www.youtube.com/@um, https://x/@dois")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    from mc.ingest.discover import load_channels

    assert load_channels() == (
        "https://www.youtube.com/@um",
        "https://x/@dois",
    )


def test_malformed_output_clear_error(monkeypatch):
    monkeypatch.setattr(
        "mc.ingest.discover._run_ytdlp", _fake_run(0, "isto não é json {", "")
    )
    with pytest.raises(DiscoveryError, match="flat-playlist"):
        list_channel_videos(["https://www.youtube.com/@x"])


def test_malformed_entries_clear_error(monkeypatch):
    stdout = json.dumps({"entries": ["not-a-dict", None]})
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(0, stdout, ""))
    with pytest.raises(DiscoveryError):
        list_channel_videos(["https://www.youtube.com/@x"])


def test_ytdlp_failure_raises(monkeypatch):
    monkeypatch.setattr("mc.ingest.discover._run_ytdlp", _fake_run(1, "", "boom"))
    with pytest.raises(DiscoveryError, match="boom"):
        list_channel_videos(["https://www.youtube.com/@x"])
