"""Channel video discovery (Story 3.1): yt-dlp --flat-playlist listing.

Channel list is configuration (env MC_CHANNELS / config.yaml) — never
hardcoded. Output parsing is strict: malformed yt-dlp output is a clear
DiscoveryError, not a silent empty set.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess

logger = logging.getLogger(__name__)

#: Only sane YouTube-ish ids become path components / URL fragments.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_YTDLP_TIMEOUT_S = 120


class DiscoveryError(RuntimeError):
    pass


def load_channels() -> tuple[str, ...]:
    """Channels from config.yaml, falling back to the MC_CHANNELS env."""
    import os

    import yaml

    from mc.config import PROJECT_ROOT

    cfg = PROJECT_ROOT / "config.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        channels = data.get("channels")
        if channels:
            return tuple(str(c) for c in channels)
    raw = os.environ.get("MC_CHANNELS", "")
    return tuple(c.strip() for c in raw.split(",") if c.strip())


def _run_ytdlp(channel_url: str) -> subprocess.CompletedProcess:
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-single-json",
        "--no-warnings",
        "--playlist-end",
        "50",
        channel_url,
    ]
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=_YTDLP_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as exc:
        raise DiscoveryError(
            f"yt-dlp timed out after {_YTDLP_TIMEOUT_S}s for {channel_url}"
        ) from exc


def list_channel_videos(channels: tuple[str, ...] | list[str]) -> set[str]:
    """Current video IDs across all configured channels."""
    videos: set[str] = set()
    for channel in channels:
        result = _run_ytdlp(channel)
        if result.returncode != 0:
            raise DiscoveryError(
                f"yt-dlp failed for {channel}: {result.stderr.strip()}"
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise DiscoveryError(
                f"yt-dlp --flat-playlist output for {channel} is not JSON: "
                f"{result.stdout[:200]!r}"
            ) from exc
        entries = data.get("entries")
        if entries is None:
            raise DiscoveryError(
                f"flat-playlist output for {channel} has no 'entries' key"
            )
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id"):
                raise DiscoveryError(
                    f"malformed playlist entry for {channel}: {entry!r}"
                )
            video_id = str(entry["id"])
            if not _VIDEO_ID_RE.fullmatch(video_id):
                # Non-YouTube extractors / malformed entries must never reach
                # f"{video_id}.srt" path building (traversal via '/' or '..').
                logger.warning(
                    "skipping entry with invalid video id for %s: %r",
                    channel,
                    video_id,
                )
                continue
            videos.add(video_id)
        logger.info("channel %s: %d videos", channel, len(entries))
    return videos
