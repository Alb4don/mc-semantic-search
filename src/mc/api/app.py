"""FastAPI application factory.

Story 0.2 added the liveness route; Story 4.5 brings GET /search and
Story 4.6 the freshness-aware /health. Everything is dependency-injected
(qdrant client, state store, freshness threshold) so tests can build the
app against local qdrant, unreachable clients or tmp state stores.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from mc.api.health import health_snapshot
from mc.config import (
    COLLECTION_NAME,
    FRESHNESS_MAX_HOURS,
    INGEST_DB_PATH,
    QDRANT_URL,
)
from mc.search.engine import SearchResult
from mc.search.engine import search as run_search
from mc.search.query import parse_query
from mc.search.ranking import apply_boost
from mc.search.snippets import build_snippet
from mc.state import StateStore

# Story 4.5: one request fetches at most 50 hits — bigger client limits are
# clamped (never rejected) so no caller can demand the whole archive.
MAX_SEARCH_LIMIT = 50


def _serialize(result: SearchResult, query: str) -> dict:
    # Epic field order: title,url,start_seconds,date,episode,channel,snippet,score.
    # SearchResult carries no video title; `episode` is the human-facing label,
    # so it doubles as the title (fallback: "vídeo {video_id}").
    return {
        "title": result.episode or f"vídeo {result.video_id}",
        "url": result.url,
        "start_seconds": result.start_seconds,
        "date": result.date,
        "episode": result.episode,
        "channel": result.channel,
        "snippet": build_snippet(result.text or "", query).highlighted,
        "score": result.score,
    }


def create_app(
    *,
    qdrant_url: str = QDRANT_URL,
    collection: str = COLLECTION_NAME,
    state_path: Path = INGEST_DB_PATH,
    freshness_max_hours: int = FRESHNESS_MAX_HOURS,
    client: QdrantClient | None = None,
    state: StateStore | None = None,
    cache: object | None = None,
) -> FastAPI:
    qdrant = client if client is not None else QdrantClient(url=qdrant_url)
    # StateStore holds a thread-bound sqlite connection: keep only the path and
    # open a short-lived store per request (same pattern as ResultCache).
    _state_db = Path(state.path) if state is not None else Path(state_path)

    @contextmanager
    def _state():
        store = StateStore(_state_db)
        try:
            yield store
        finally:
            store.close()

    app = FastAPI(title="mc-semantic-search", version="0.1.0")
    # Story 4.7 (ResultCache) will plug in here: the cache is an embedder-level
    # LRU and the engine embeds internally, so the wiring lands in engine.py.
    app.state.cache = cache

    def _freshness() -> dict:
        try:
            with _state() as store:
                return {"last_update": store.last_success("update")}
        except Exception:  # unreadable state must never break search
            return {"last_update": None}

    @app.get("/search")
    def search_endpoint(
        query: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1),
    ) -> dict:
        effective_limit = min(limit, MAX_SEARCH_LIMIT)
        started = time.perf_counter()
        try:
            results: list[SearchResult] = run_search(
                qdrant, query, limit=effective_limit, collection=collection
            )
        except Exception as exc:  # transport failure → 503, never 500
            raise HTTPException(
                status_code=503, detail="qdrant unreachable"
            ) from exc
        # Story 4.3: verbatim-phrase queries demote non-literal hits so the
        # exact moment surfaces first (never dropped, count unchanged).
        results = apply_boost(results, parse_query(query))
        took_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "took_ms": took_ms,
            "freshness": _freshness(),
            "results": [_serialize(r, query) for r in results],
        }

    @app.get("/health")
    def health_endpoint() -> JSONResponse:
        with _state() as store:
            code, body = health_snapshot(
                qdrant, store, freshness_max_hours, collection=collection
            )
        return JSONResponse(status_code=code, content=body)

    return app


app = create_app()
