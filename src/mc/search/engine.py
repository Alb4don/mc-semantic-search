"""Hybrid search: dense + sparse prefetch fused by Qdrant's native RRF.

Story 4.2. The engine owns fusion — the API layer only receives a typed
list of results. No boosting/caching here (Story 4.3 extends this).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from mc.config import COLLECTION_NAME
from mc.embeddings import embed_query, embed_sparse
from mc.qdrant import DENSE_VECTOR, SPARSE_VECTOR
from mc.search.query import parse_query

# NFR-1: one embedding call at a time on a 2-4 vCPU VPS.
EMBED_EXECUTOR_WORKERS = 1


@dataclass(frozen=True)
class SearchResult:
    """One transcript chunk, ranked by RRF-fused hybrid retrieval."""

    video_id: str
    text: str
    url: str
    start_seconds: int
    date: str | None
    episode: str | None
    channel: str | None
    score: float


def search(
    client: QdrantClient,
    query_raw: str,
    *,
    limit: int = 10,
    collection: str = COLLECTION_NAME,
) -> list[SearchResult]:
    """Run hybrid (dense+sparse, RRF-fused) search on the given collection."""
    query = parse_query(query_raw)
    if not query.normalized:
        return []

    # NFR-1: both embedder calls inside a max_workers=1 executor.
    with ThreadPoolExecutor(max_workers=EMBED_EXECUTOR_WORKERS) as pool:
        dense_future = pool.submit(embed_query, query.normalized)
        sparse_future = pool.submit(embed_sparse, [query.normalized])
        dense = dense_future.result()
        (sparse,) = sparse_future.result()

    prefetch: list[models.Prefetch] = [
        models.Prefetch(query=dense, using=DENSE_VECTOR, limit=max(limit, 10))
    ]
    if sparse.indices:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=list(sparse.indices),
                    values=list(sparse.values),
                ),
                using=SPARSE_VECTOR,
                limit=max(limit, 10),
            )
        )

    response = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [_to_result(p) for p in response.points if not p.payload.get("__meta__")]


def _to_result(point) -> SearchResult:
    payload = point.payload
    return SearchResult(
        video_id=payload.get("video_id"),
        text=payload.get("text"),
        url=payload.get("url"),
        start_seconds=payload.get("start_seconds"),
        date=payload.get("date"),
        episode=payload.get("episode"),
        channel=payload.get("channel"),
        score=float(point.score),
    )
