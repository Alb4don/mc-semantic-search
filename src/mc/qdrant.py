"""Qdrant collection management and atomic per-video upserts.

Stories 1.3/1.4. The collection stores two named vectors (`dense` e5,
`sparse` bm25) plus payload indexes; the model fingerprint lives in a
sentinel meta point so a model swap is rejected loudly (NFR-6).
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import QdrantClient, models

from mc.config import DENSE_DIM, QDRANT_API_KEY
from mc.embeddings import embed_passage, embed_sparse, model_fingerprint, sparse_fingerprint
from mc.ingest.models import Chunk

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
COLLECTION_META_POINT_ID = 0  # sentinel point holding the model fingerprint
KEYWORD_INDEX_FIELDS = ["video_id", "channel"]
DATETIME_INDEX_FIELDS = ["date"]


class ModelMismatchError(RuntimeError):
    """Raised when the collection fingerprint doesn't match the loaded models."""


def client_kwargs(*, timeout: float | None = None) -> dict:
    """Shared QdrantClient kwargs: api_key attached when configured.

    Defense in depth for an auth-enabled qdrant server — config.QDRANT_API_KEY
    is OFF by default; pair it with QDRANT__SERVICE__API_KEY server-side
    (see docker-compose.yml). New QdrantClient(...) call sites should spread
    these kwargs.
    """
    kwargs: dict = {"timeout": timeout} if timeout is not None else {}
    if QDRANT_API_KEY:
        kwargs["api_key"] = QDRANT_API_KEY
    return kwargs


def ensure_collection(client: QdrantClient, name: str, *, force: bool = False) -> None:
    """Create the collection if missing; validate fingerprint if present."""
    exists = True
    try:
        client.get_collection(name)
        _validate_fingerprint(client, name)
        return
    except ModelMismatchError:
        if not force:
            raise
        logger.warning("fingerprint mismatch — force recreating %s", name)
        client.delete_collection(name)
        exists = False
    except Exception as exc:
        if "Not found" not in str(exc) and "404" not in str(exc):
            raise
        exists = False

    if exists:
        return
    fingerprint = {"dense": model_fingerprint(), "sparse": sparse_fingerprint()}
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                index=models.SparseIndexParams(),
            )
        },
    )
    for field in KEYWORD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    for field in DATETIME_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=models.PayloadSchemaType.DATETIME,
        )
    _write_meta_fingerprint(client, name, fingerprint)
    logger.info("collection %s created (fingerprint recorded)", name)


def _write_meta_fingerprint(client: QdrantClient, name: str, fingerprint: dict) -> None:
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=COLLECTION_META_POINT_ID,
                vector={DENSE_VECTOR: [0.0] * DENSE_DIM},
                payload={"__meta__": True, "fingerprint": fingerprint},
            )
        ],
    )


def read_meta_fingerprint(client: QdrantClient, name: str) -> dict:
    points = client.retrieve(
        collection_name=name,
        ids=[COLLECTION_META_POINT_ID],
        with_payload=True,
    )
    if not points:
        raise ModelMismatchError(f"collection {name} has no fingerprint meta point")
    return points[0].payload["fingerprint"]


def _validate_fingerprint(client: QdrantClient, name: str) -> None:
    stored = read_meta_fingerprint(client, name)
    current = {"dense": model_fingerprint(), "sparse": sparse_fingerprint()}
    if stored.get("dense") != current["dense"]:
        raise ModelMismatchError(
            f"dense model mismatch for {name}: stored={stored['dense']} "
            f"current={current['dense']} — run `mc reindex` instead of "
            "writing into a different model space"
        )
    if stored.get("sparse") != current["sparse"]:
        raise ModelMismatchError(
            f"sparse encoder mismatch for {name}: stored={stored['sparse']}"
        )


def get_collection(client: QdrantClient, name: str):
    return client.get_collection(name)


def upsert_video(
    client: QdrantClient,
    collection_name: str,
    video_id: str,
    chunks: list[Chunk],
    *,
    force: bool = False,
    embed_batch_size: int | None = None,
) -> None:
    """Replace the video's points in ONE atomic operation (Story 1.4, NFR-3).

    Delete-then-insert semantics inside a single Qdrant write batch: a
    crash can never leave half a video indexed. Embedding may be split
    into `embed_batch_size`-sized calls; the commit stays atomic.
    """
    ensure_collection(client, collection_name, force=force)

    texts = [c.text for c in chunks]
    if embed_batch_size:
        dense: list[list[float]] = []
        sparse: list = []
        for i in range(0, len(texts), embed_batch_size):
            dense.extend(embed_passage(texts[i : i + embed_batch_size]))
            sparse.extend(embed_sparse(texts[i : i + embed_batch_size]))
    else:
        dense = embed_passage(texts)
        sparse = embed_sparse(texts)

    operations: list = []
    old_ids = _existing_point_ids(client, collection_name, video_id)
    if old_ids:
        operations.append(
            models.DeleteOperation(delete=models.PointIdsList(points=old_ids))
        )
    operations.append(
        models.UpsertOperation(
            upsert=models.PointsList(
                points=[
                    models.PointStruct(
                        id=_point_id(video_id, c.start_seconds),
                        vector={
                            DENSE_VECTOR: dense[i],
                            SPARSE_VECTOR: models.SparseVector(
                                indices=list(sparse[i].indices),
                                values=list(sparse[i].values),
                            ),
                        },
                        payload=c.to_payload(),
                    )
                    for i, c in enumerate(chunks)
                ]
            )
        )
    )
    client.batch_update_points(
        collection_name=collection_name, update_operations=operations
    )


def _existing_point_ids(client: QdrantClient, name: str, video_id: str) -> list[int]:
    ids: list[int] = []
    offset = None
    while True:
        scrolled, next_offset = client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="video_id", match=models.MatchValue(value=video_id)
                    )
                ]
            ),
            limit=1000,
            offset=offset,
        )
        ids.extend(p.id for p in scrolled)
        if next_offset is None:
            return ids
        offset = next_offset


def _point_id(video_id: str, start_seconds: int) -> int:
    """Stable point id: same video+timestamp → same id (idempotent upserts)."""
    digest = uuid.uuid5(uuid.NAMESPACE_URL, f"{video_id}:{start_seconds}")
    return int(digest.hex[:16], 16)
