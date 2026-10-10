"""Backfill orchestrator (Story 2.4): embed + index the legacy archive.

Resumable (NFR-3): checkpoints are the per-video `processed` markers in
state/ingest.db, written only AFTER the per-video commit succeeds — a
crash at hour 2 resumes, never restarts. Consecutive rows of the legacy
DB for the same video are grouped so each video commits atomically via
upsert_video (Story 1.4).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from mc.config import COLLECTION_NAME, INGEST_DB_PATH, QDRANT_URL
from mc.ingest.legacy import count_legacy, iter_legacy_chunks
from mc.ingest.lock import RunLock
from mc.qdrant import ensure_collection, upsert_video
from mc.state import StateStore

logger = logging.getLogger(__name__)


class BackfillInterrupted(RuntimeError):
    """Raised by the kill_after_commit hook to simulate a crash."""


@dataclass
class BackfillReport:
    indexed_chunks: int = 0
    quarantined: int = 0
    skipped_videos: int = 0
    videos_processed: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"backfill: {self.indexed_chunks} chunks indexed, "
            f"{self.videos_processed} videos, "
            f"{self.quarantined} quarantined, "
            f"{self.skipped_videos} videos skipped (already processed)"
        )


def run_backfill(
    from_db,
    batch_size: int = 256,
    limit: int | None = None,
    *,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    state: StateStore | None = None,
    quarantine_path=None,
    lock_path=None,
    kill_after_commit=None,
) -> dict:
    """Embed + index all unprocessed videos from the legacy DB.

    batch_size caps the texts per embedding call inside upsert; commits
    and checkpoints are per video. `limit` caps the number of videos
    processed this run.
    """
    if client is None:
        client = QdrantClient(url=QDRANT_URL, timeout=60)
    if state is None:
        state = StateStore(INGEST_DB_PATH)

    lock = RunLock(lock_path or _default_lock())
    report = BackfillReport()

    counts = count_legacy(from_db)
    report.notes.append(f"legacy db: {counts}")
    report.quarantined = int(counts.get("quarantined", 0))

    with lock:
        ensure_collection(client, collection)
        processed = 0
        for video_id, chunks in _group_by_video(from_db, quarantine_path):
            if limit is not None and processed >= limit:
                break
            if state.is_processed(video_id):
                report.skipped_videos += 1
                logger.debug("skipping %s (already processed)", video_id)
                continue
            logger.info(
                "video %s: embedding + indexing %d chunks",
                video_id,
                len(chunks),
            )
            upsert_video(
                client, collection, video_id, chunks, embed_batch_size=batch_size
            )
            state.mark_processed(video_id, "backfill")
            report.indexed_chunks += len(chunks)
            report.videos_processed += 1
            processed += 1
            if kill_after_commit is not None:
                kill_after_commit(video_id)  # may raise BackfillInterrupted

    logger.info(report.summary())
    return {
        "indexed_chunks": report.indexed_chunks,
        "quarantined": report.quarantined,
        "skipped_videos": report.skipped_videos,
        "videos_processed": report.videos_processed,
        "notes": report.notes,
    }

def _default_lock():
    from mc.config import DATA_DIR

    return DATA_DIR / "update.lock"


def _group_by_video(from_db, quarantine_path):
    """Yield (video_id, [Chunk, ...]) preserving legacy row order."""
    for video_id, rows in itertools.groupby(
        iter_legacy_chunks(from_db, quarantine_path=quarantine_path),
        key=lambda c: c.video_id,
    ):
        yield video_id, list(rows)
