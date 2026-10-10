"""Story 2.2 — yt-dlp metadata enrichment with a resumable on-disk cache.

One JSON file per video under the metadata cache dir (config.META_CACHE_DIR by
default), so a re-run never re-invokes yt-dlp for an id it already enriched.
Videos that fail enrichment after the retries end up in ``missing.txt`` and are
simply omitted from the returned mapping; the rest of the batch continues.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from mc import config

Fetcher = Callable[[str], dict]

_YOUTUBE_URL = "https://www.youtube.com/watch?v={video_id}"
_EPISODE_RE = r"#(\d+)"
_FALLBACK_DIGITS_RE = r"\d+"
_MAX_ATTEMPTS = 3


def _default_fetch(video_id: str) -> dict:
    """Fetch raw metadata for one video via the pinned yt-dlp dependency."""
    url = _YOUTUBE_URL.format(video_id=video_id)
    proc = subprocess.run(
        ["yt-dlp", "--dump-single-json", "--no-warnings", url],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def _parse_episode(title: str) -> str | None:
    """Episode number from the title: prefer ``#N``, else the last digit run."""
    match = re.search(_EPISODE_RE, title)
    if match:
        return match.group(1)
    digits = re.findall(_FALLBACK_DIGITS_RE, title)
    return digits[-1] if digits else None


def _normalize(raw: dict) -> dict:
    upload_date = raw.get("upload_date") or ""
    try:
        date = datetime.strptime(upload_date, "%Y%m%d").date().isoformat()
    except ValueError:
        date = None
    return {
        "date": date,
        "episode": _parse_episode(raw.get("title") or ""),
        "channel": raw.get("uploader") or raw.get("channel"),
    }


def enrich(
    video_ids: list[str],
    *,
    cache_dir: Path | None = None,
    fetch: Fetcher | None = None,
    backoff: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> dict[str, dict]:
    """Enrich video ids with {date, episode, channel}, cached on disk.

    Duplicated ids are fetched once; ids already cached are loaded from disk
    without invoking the fetcher. Fetch failures are retried with exponential
    backoff (``backoff`` delays); after the final failure the id is appended to
    ``missing.txt`` and omitted from the result.
    """
    cache = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    fetcher = fetch if fetch is not None else _default_fetch
    cache.mkdir(parents=True, exist_ok=True)

    result: dict[str, dict] = {}
    for video_id in dict.fromkeys(video_ids):
        cache_file = cache / f"{video_id}.json"
        if cache_file.exists():
            result[video_id] = json.loads(cache_file.read_text(encoding="utf-8"))
            continue

        raw: dict | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                raw = fetcher(video_id)
                break
            except Exception:
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(backoff[min(attempt, len(backoff) - 1)])
        if raw is None:
            with (cache / "missing.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"{video_id}\n")
            continue

        meta = _normalize(raw)
        cache_file.write_text(json.dumps(meta), encoding="utf-8")
        result[video_id] = meta
    return result
