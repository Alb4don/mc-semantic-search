"""One-off archive-stats backfill (Story 6.4).

Fills ``video_stats`` for the videos already indexed, so the landing-page
"status do acervo" shows full numbers from day one. Resumable and idempotent:
a video whose stats row already has a duration is skipped, and a fetched cache
entry is never re-fetched.

Duration precedence (highest wins, fills NULLs only):
  ① metadata cache entry (arrives via the ingest pipeline)
  ② local transcripts cache ``transcripts/<id>.srt`` — ``max(start+duration)``
     over the JSON segment list (the reference-script technique, zero network)
  ③ with ``--fetch-missing`` only: one yt-dlp ``--dump-single-json`` fetch,
     whose result also upgrades the cache file with the duration (once ever)

No fetch happens by default, so a run over ~3.5k videos is seconds of local
JSON parsing and never exposes a datacenter IP to YouTube bot detection.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mc import config
from mc.ingest.metadata import _default_fetch, _normalize, episode_int
from mc.state import StateStore

logger = logging.getLogger(__name__)

#: Seconds to sleep between yt-dlp fetches when --fetch-missing is in effect.
_FETCH_BACKOFF_S = 5.0


@dataclass
class StatsBackfillReport:
    videos_considered: int = 0
    videos_backfilled: int = 0  # gained a non-null duration this run
    videos_skipped: int = 0  # already complete — resumable no-op
    still_missing: int = 0  # duration still None after the run
    fetched: int = 0  # durations obtained via a yt-dlp fetch
    notes: list[str] = field(default_factory=list)

    def summary(self, stats: dict) -> str:
        seconds = int(stats.get("seconds") or 0)
        return (
            f"stats-backfill: {self.videos_backfilled} backfilled, "
            f"{self.videos_skipped} already complete, "
            f"{self.still_missing} still missing duration; "
            f"{stats['videos']} videos, {seconds} s "
            f"({seconds // 60} min, {seconds // 3600} h, {seconds // 86400} d)"
        )


def _transcript_duration(segments: list[dict]) -> int | None:
    """``int(max(start + duration))`` over the transcript segments (the
    reference-script technique). None when there are no usable segments."""
    ends: list[float] = []
    for seg in segments:
        try:
            ends.append(float(seg["start"]) + float(seg["duration"]))
        except (KeyError, TypeError, ValueError):
            continue
    return int(max(ends)) if ends else None


def _load_transcript(transcripts_dir: Path, video_id: str) -> list[dict] | None:
    """The cached transcript segment list, or None when absent/unreadable."""
    path = transcripts_dir / f"{video_id}.srt"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("transcript for %s unreadable (%s)", video_id, exc)
        return None
    return data if isinstance(data, list) else None


def _load_cache(cache_dir: Path, video_id: str) -> dict:
    path = cache_dir / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _cache_ids(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    return {p.stem for p in cache_dir.glob("*.json")}


def _resolve_duration(
    video_id: str,
    cache: dict,
    cache_dir: Path,
    transcripts_dir: Path,
    *,
    fetch_missing: bool,
    fetch: Callable[[str], dict] | None,
    sleeper: Callable[[float], None],
) -> tuple[int | None, bool]:
    """Return (duration, fetched). Precedence cache > transcripts > fetch.

    A fetch (only when ``fetch_missing``) upgrades the cache file with the
    duration so it is fetched at most once ever.
    """
    if cache.get("duration") is not None:
        return int(cache["duration"]), False

    segments = _load_transcript(transcripts_dir, video_id)
    if segments is not None:
        derived = _transcript_duration(segments)
        if derived is not None:
            return derived, False

    if fetch_missing and fetch is not None:
        try:
            raw = fetch(video_id)
        except Exception as exc:  # noqa: BLE001 — skip, never fatal
            logger.warning("fetch failed for %s: %s", video_id, exc)
            return None, False
        sleeper(_FETCH_BACKOFF_S)
        meta = _normalize(raw)  # includes duration; upgrades the cache entry
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{video_id}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
        if meta.get("duration") is not None:
            return int(meta["duration"]), True

    return None, False


def run_stats_backfill(
    *,
    fetch_missing: bool = False,
    limit: int | None = None,
    state: StateStore | None = None,
    cache_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    fetch: Callable[[str], dict] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict:
    """Backfill ``video_stats`` for the already-indexed archive.

    Video universe = ``state.processed`` ∪ metadata-cache ids. Per video,
    episode/date come from the cache and duration follows the precedence above.
    ``limit`` bounds the number of videos considered this run. Returns a report
    dict; the operator-facing summary is printed by the CLI.
    """
    state = state or StateStore(config.INGEST_DB_PATH)
    cache_dir = Path(cache_dir) if cache_dir is not None else config.META_CACHE_DIR
    transcripts_dir = (
        Path(transcripts_dir) if transcripts_dir is not None else config.TRANSCRIPTS_DIR
    )
    fetcher = fetch if fetch is not None else (
        _default_fetch if fetch_missing else None
    )

    report = StatsBackfillReport()
    universe = sorted(set(state.processed_ids()) | _cache_ids(cache_dir))

    for video_id in universe:
        if limit is not None and report.videos_considered >= limit:
            break
        report.videos_considered += 1

        # Resumable: a video that already has a duration is done — skip the
        # (possibly expensive) resolution entirely, and never re-fetch it.
        if not fetch_missing and state.video_duration(video_id) is not None:
            report.videos_skipped += 1
            continue

        cache = _load_cache(cache_dir, video_id)
        duration, fetched = _resolve_duration(
            video_id,
            cache,
            cache_dir,
            transcripts_dir,
            fetch_missing=fetch_missing,
            fetch=fetcher,
            sleeper=sleeper,
        )
        if fetched:
            report.fetched += 1
        episode = episode_int(cache.get("episode"))
        date = cache.get("date") or None

        try:
            state.record_video_stats(
                video_id, duration=duration, episode=episode, date=date
            )
        except Exception:  # noqa: BLE001 — never fatal to the backfill
            logger.warning("stats write failed for %s", video_id, exc_info=True)
            continue

        if duration is not None:
            report.videos_backfilled += 1
        else:
            report.still_missing += 1

    stats = state.archive_stats()
    logger.info(report.summary(stats))
    return {
        "videos_considered": report.videos_considered,
        "videos_backfilled": report.videos_backfilled,
        "videos_skipped": report.videos_skipped,
        "still_missing": report.still_missing,
        "fetched": report.fetched,
        "videos": stats["videos"],
        "seconds": stats["seconds"],
        "summary": report.summary(stats),
    }
