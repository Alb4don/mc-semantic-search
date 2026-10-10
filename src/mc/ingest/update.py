"""Daily ingestion orchestrator (Story 3.4).

Discover → download → chunk → embed → upsert → mark processed, one
atomic per-video transaction at a time. Chunk records are staged to
``data/queue/pending.jsonl`` so ``mc push``/``mc apply`` (Mode B) ship
exactly the same increment that Mode A indexes directly.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable
from pathlib import Path

from qdrant_client import QdrantClient

from mc.config import (
    COLLECTION_NAME,
    COOKIES_PATH,
    DATA_DIR,
    META_CACHE_DIR,
    QDRANT_URL,
    TRANSCRIPTS_DIR,
)
from mc.ingest.chunker import chunk_segments
from mc.ingest.discover import list_channel_videos, load_channels
from mc.ingest.download import (
    IGNORE_LIST_FILENAME,
    add_to_ignore_list,
    download_transcript,
)
from mc.ingest.lock import RunLock
from mc.ingest.metadata import enrich
from mc.qdrant import ensure_collection, upsert_video
from mc.state import StateStore

logger = logging.getLogger(__name__)

PENDING_NAME = "pending.jsonl"

# Story 5.6: stale cookies look like any other download failure to yt-dlp,
# but ignore-listing the video would skip it forever. Auth-shaped errors are
# reported as a distinct reason and never ignore-list.
_AUTH_MARKERS = (
    "sign in",
    "sign-in",
    "log in",
    "login",
    "cookies",
    "confirm you",
    "not a bot",
    "membership",
    "unauthorized",
)


def is_auth_failure(reason: str) -> bool:
    lowered = (reason or "").lower()
    return any(marker in lowered for marker in _AUTH_MARKERS)


def run_update(
    *,
    dry_run: bool = False,
    channels: tuple[str, ...] | None = None,
    state: StateStore | None = None,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    discover: Callable[[tuple[str, ...]], list[str] | set[str]] | None = None,
    downloader: Callable[[str], list[dict] | None] | None = None,
    metadata_fetch: Callable[[list[str]], dict] | None = None,
    lock_path: Path | None = None,
    cookies_path: str | None = None,
    transcripts_dir: Path | None = None,
    queue_dir: Path | None = None,
) -> int:
    """Run one daily ingestion pass. Returns a process exit code."""
    state = state or StateStore(DATA_DIR / "ingest.db")
    client = client or QdrantClient(url=QDRANT_URL, timeout=60)
    channels = channels or load_channels()
    discover = discover or list_channel_videos
    downloader = downloader or _default_downloader(cookies_path, transcripts_dir)
    transcripts_dir = Path(transcripts_dir or TRANSCRIPTS_DIR)
    queue_dir = Path(queue_dir or DATA_DIR / "queue")
    lock_path = lock_path or DATA_DIR / "update.lock"

    discovered = list(discover(channels))
    new_ids = sorted(vid for vid in discovered if not state.is_processed(vid))

    if dry_run:
        logger.info("dry run: %d new videos would be ingested", len(new_ids))
        return 0

    with RunLock(lock_path):
        ensure_collection(client, collection)
        metadata = (
            metadata_fetch(new_ids)
            if metadata_fetch is not None
            else enrich(new_ids, cache_dir=META_CACHE_DIR)
        )

        failures: list[tuple[str, str]] = []
        for video_id in new_ids:
            try:
                _ingest_one(
                    client, collection, video_id, downloader, metadata, state, queue_dir
                )
            except Exception as exc:  # noqa: BLE001 — isolate every video
                reason = str(exc) or type(exc).__name__
                if is_auth_failure(reason):
                    logger.error(
                        "auth failure for %s: cookies likely stale — "
                        "rotate them (docs/tasks/cookies.md)",
                        video_id,
                    )
                    failures.append((video_id, f"auth: {reason}"))
                    continue
                logger.exception("video %s failed", video_id)
                failures.append((video_id, reason))
                add_to_ignore_list(
                    video_id, transcripts_dir / IGNORE_LIST_FILENAME, reason
                )

        logger.info(
            "run summary: %d new videos, %d indexed, %d failed",
            len(new_ids),
            len(new_ids) - len(failures),
            len(failures),
        )
        if failures:
            for video_id, reason in failures:
                logger.warning("failed video %s: %s", video_id, reason)
            return 1

        state.record_success("update")
        return 0


def _ingest_one(
    client: QdrantClient,
    collection: str,
    video_id: str,
    downloader: Callable[[str], list[dict] | None],
    metadata: dict[str, dict],
    state: StateStore,
    queue_dir: Path,
) -> None:
    """Download → chunk → upsert → mark processed for ONE video."""
    segments = downloader(video_id)
    if segments is None:
        raise LookupError("no subtitles available in any supported language")

    info = metadata.get(video_id, {})
    chunks = [
        dataclasses.replace(
            chunk,
            date=info.get("date"),
            episode=info.get("episode"),
            channel=info.get("channel"),
        )
        for chunk in chunk_segments(segments, video_id)
    ]
    upsert_video(client, collection, video_id, chunks)
    state.mark_processed(video_id, "update")
    _stage_for_push(chunks, queue_dir)


def _stage_for_push(chunks, queue_dir: Path) -> None:
    """Append the chunk records to data/queue/pending.jsonl (Mode B increment)."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    with open(queue_dir / PENDING_NAME, "a", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_payload(), sort_keys=True) + "\n")


def _default_downloader(cookies_path: str | None, transcripts_dir: Path | None):
    cookies = cookies_path or COOKIES_PATH
    dir_ = transcripts_dir or TRANSCRIPTS_DIR

    def download(video_id: str) -> list[dict] | None:
        return download_transcript(
            video_id, cookies_path=cookies, transcripts_dir=Path(dir_)
        )

    return download
