"""`mc reindex` — model-swap migration with atomic alias switch (Story 3.8, NFR-6).

Rebuilds the whole index into a NEW collection by re-reading the chunk
payloads from the source collection (no re-transcription), re-embedding
with the currently-loaded embedder, and switching the alias to the new
collection in ONE atomic Qdrant alias operation. The old collection is
never deleted, so rollback is re-pointing the alias by hand.

Design notes / limitations:
- `--model` is accepted and recorded in the report, but the model swap
  itself is a bake-time operation (models live in the image and are
  fingerprinted into collections, NFR-6): this run always re-embeds with
  the currently-loaded embedder. When `model` differs from the loaded
  id, a note is added to the report instead of failing.
- Refusal logic is the NFR-6 guard: a target collection whose stored
  fingerprint doesn't match the embedder about to be used is NEVER
  written into, and a same-fingerprint target is refused too — the
  operator drops it first. Resuming into a partially-built collection is
  intentionally out of scope (the run is cheap relative to a re-scrape).
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient, models

from mc.config import COLLECTION_NAME, DENSE_MODEL_ID, QDRANT_URL
from mc.embeddings import model_fingerprint, sparse_fingerprint
from mc.ingest.models import Chunk
from mc.qdrant import (
    COLLECTION_META_POINT_ID,
    ModelMismatchError,
    client_kwargs,
    ensure_collection,
    read_meta_fingerprint,
    upsert_video,
)

logger = logging.getLogger(__name__)


class ReindexError(RuntimeError):
    """Operator-facing refusal: reindex preconditions not met."""


def run_reindex(
    *,
    model: str | None = None,
    client: QdrantClient | None = None,
    source_collection: str = COLLECTION_NAME,
    target_collection: str | None = None,
) -> dict:
    """Rebuild `source_collection` into `target_collection` and swap the alias.

    Steps: refuse bad targets -> scroll source payloads (skipping the
    meta sentinel) -> create target (fingerprinted) -> upsert every video
    (re-embeds) -> move the alias in ONE atomic alias update. Returns a
    report dict with ``points_copied``, ``videos``, ``target``, ``alias``
    and model bookkeeping.
    """
    if client is None:
        client = QdrantClient(url=QDRANT_URL, **client_kwargs(timeout=60))
    target = target_collection or f"{source_collection}-v2"

    _refuse_bad_target(client, target)

    by_video, points_copied = _read_source_chunks(client, source_collection)

    ensure_collection(client, target)
    for video_id, chunks in sorted(by_video.items()):
        upsert_video(client, target, video_id, chunks)

    alias = _swap_alias(client, source_collection, target)
    logger.info(
        "reindexed %s points / %s videos into %s; alias %s swapped atomically",
        points_copied,
        len(by_video),
        target,
        alias,
    )

    report: dict = {
        "points_copied": points_copied,
        "videos": len(by_video),
        "target": target,
        "alias": alias,
        "model_requested": model,
        "model_used": DENSE_MODEL_ID,
    }
    if model and model != DENSE_MODEL_ID:
        report["notes"] = [
            f"requested model {model!r} != loaded {DENSE_MODEL_ID!r}: re-embedded "
            "with the loaded model — actual model swaps are bake-time operations "
            "(NFR-6); drop the old collection only after validating searches"
        ]
    return report


def _refuse_bad_target(client: QdrantClient, target: str) -> None:
    """Refuse any pre-existing target (NFR-6 fingerprint guard).

    A different fingerprint means the target belongs to another model
    space; a matching one means a previous run's leftover — rebuilding
    into it would mix generations, so both cases refuse.
    """
    try:
        client.get_collection(target)
    except Exception as exc:
        if "Not found" not in str(exc) and "404" not in str(exc):
            raise
        return

    try:
        stored = read_meta_fingerprint(client, target)
    except ModelMismatchError:
        raise ReindexError(
            f"target collection {target!r} exists but has no readable "
            "fingerprint meta point — drop it first or pick another target"
        ) from None
    expected = {"dense": model_fingerprint(), "sparse": sparse_fingerprint()}
    if stored != expected:
        raise ReindexError(
            f"target collection {target!r} exists with a different fingerprint: "
            f"stored={stored} expected={expected} — refusing to rebuild into "
            "another model's collection"
        )
    raise ReindexError(
        f"target collection {target!r} already exists with a matching "
        "fingerprint — drop it first to rebuild from scratch"
    )


def _read_source_chunks(client: QdrantClient, source: str) -> tuple[dict[str, list[Chunk]], int]:
    """Scroll ALL source points except the meta sentinel, grouped by video_id."""
    by_video: dict[str, list[Chunk]] = {}
    points_copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            if point.id == COLLECTION_META_POINT_ID:
                continue
            payload = point.payload or {}
            video_id = payload.get("video_id")
            if video_id is None:
                logger.warning("skipping point %s without video_id", point.id)
                continue
            by_video.setdefault(str(video_id), []).append(
                Chunk(
                    video_id=str(video_id),
                    text=payload["text"],
                    url=payload["url"],
                    start_seconds=payload["start_seconds"],
                    date=payload.get("date"),
                    episode=payload.get("episode"),
                    channel=payload.get("channel"),
                )
            )
            points_copied += 1
        if offset is None:
            return by_video, points_copied


def _swap_alias(client: QdrantClient, source: str, target: str) -> str:
    """Point the alias at `target` in ONE atomic alias update (never deletes).

    If the source collection holds an alias, that name is deleted and
    re-created on the target inside a single update_collection_aliases
    call (Qdrant guarantees the batch is atomic). Otherwise a fresh
    ``<source>-alias`` alias is created on the target.
    """
    existing = _source_aliases(client, source)
    alias = existing[0] if existing else f"{source}-alias"
    operations: list = []
    if existing:
        for name in existing:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=name)
                )
            )
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=target, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    return alias


def _source_aliases(client: QdrantClient, source: str) -> list[str]:
    response = client.get_aliases()
    return sorted(a.alias_name for a in response.aliases if a.collection_name == source)
