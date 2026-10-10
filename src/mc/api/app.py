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

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

TEMPLATES = Jinja2Templates(
    directory=Path(__file__).parent / "templates", autoescape=True
)


def _tape(start_seconds: int) -> str:
    """Tape-counter rendering of the chunk offset — the design anchor."""
    total = int(start_seconds or 0)
    hours, rest = divmod(total, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"T+{hours:02d}:{minutes:02d}:{seconds:02d}"


def _freshness_line(last_update: str | None) -> str:
    """Portuguese archive-freshness line shown in the footer/empty state."""
    if not last_update:
        return "acervo atualizado agora mesmo."
    date = last_update[:10]
    return f"acervo atualizado até {date}"


def _video_result(result: SearchResult, query: str) -> dict:
    return {
        "title": result.episode or f"vídeo {result.video_id}",
        "episode": result.episode,
        "date": result.date,
        "channel": result.channel,
        "snippet": build_snippet(result.text or "", query).highlighted,
        "url": result.url,
        "video_id": result.video_id,
        "start_seconds": result.start_seconds,
        "tape": _tape(result.start_seconds),
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
    app.mount(
        "/static",
        StaticFiles(directory=Path(__file__).parent / "static"),
        name="static",
    )
    # Story 4.7 (ResultCache) will plug in here: the cache is an embedder-level
    # LRU and the engine embeds internally, so the wiring lands in engine.py.
    app.state.cache = cache

    def _freshness() -> dict:
        try:
            with _state() as store:
                return {"last_update": store.last_success("update")}
        except Exception:  # unreadable state must never break search
            return {"last_update": None}

    def _video_results(query: str, effective_limit: int) -> list[dict]:
        try:
            results: list[SearchResult] = run_search(
                qdrant, query, limit=effective_limit, collection=collection
            )
        except Exception as exc:  # transport failure → 503, never 500
            raise HTTPException(
                status_code=503, detail="qdrant unreachable"
            ) from exc
        results = apply_boost(results, parse_query(query))
        return [_video_result(r, query) for r in results]

    @app.get("/search")
    def search_endpoint(
        query: str = Query(min_length=1),
        limit: int = Query(default=10, ge=1),
    ) -> dict:
        effective_limit = min(limit, MAX_SEARCH_LIMIT)
        started = time.perf_counter()
        results = _video_results(query, effective_limit)
        took_ms = (time.perf_counter() - started) * 1000
        return {
            "query": query,
            "took_ms": took_ms,
            "freshness": _freshness(),
            "results": [
                {
                    key: r[key]
                    for key in (
                        "title",
                        "url",
                        "start_seconds",
                        "date",
                        "episode",
                        "channel",
                        "snippet",
                        "score",
                    )
                }
                for r in results
            ],
        }

    @app.get("/results", response_class=HTMLResponse)
    def results_partial(query: str = Query(default="")) -> str:
        results = _video_results(query, MAX_SEARCH_LIMIT) if query.strip() else []
        return TEMPLATES.get_template("results.html").render(
            results=results,
            query=query,
            freshness_line=_freshness_line(_freshness()["last_update"]),
        )

    @app.get("/", response_class=HTMLResponse)
    def search_page(
        request: Request, query: str = Query(default="")
    ) -> HTMLResponse:
        stripped = query.strip()
        results = _video_results(query, MAX_SEARCH_LIMIT) if stripped else []
        return TEMPLATES.TemplateResponse(
            request=request,
            name="search.html",
            context={
                "query": stripped,
                "results": results,
                "freshness_line": _freshness_line(_freshness()["last_update"]),
                "stats": "",
            },
        )

    @app.get("/health")
    def health_endpoint() -> JSONResponse:
        with _state() as store:
            code, body = health_snapshot(
                qdrant, store, freshness_max_hours, collection=collection
            )
        return JSONResponse(status_code=code, content=body)

    return app


app = create_app()
