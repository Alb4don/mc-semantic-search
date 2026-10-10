"""Story 4.7 — LRU result cache with freshness invalidation."""

from __future__ import annotations

from pathlib import Path

from mc.search.cache import ResultCache


def _spy_embedder():
    calls = []

    def embedder(query: str) -> list[float]:
        calls.append(query)
        return [float(len(query))] * 8

    return embedder, calls


def test_same_query_twice_embeds_once():
    embedder, calls = _spy_embedder()
    state_path = Path("/nonexistent/ingest.db")
    cache = ResultCache(embedder, state_path, size=8)
    cache.get("bitcoin")
    cache.get("bitcoin")
    assert len(calls) == 1


def test_different_queries_embed_separately():
    embedder, calls = _spy_embedder()
    cache = ResultCache(embedder, Path("/nonexistent"), size=8)
    cache.get("a")
    cache.get("b")
    assert len(calls) == 2


def test_cache_invalidated_when_last_success_advances(tmp_path):
    from mc.state import StateStore

    state = StateStore(tmp_path / "ingest.db")
    embedder, calls = _spy_embedder()
    cache = ResultCache(embedder, tmp_path / "ingest.db", size=8)

    cache.get("bitcoin")
    cache.get("bitcoin")
    assert len(calls) == 1

    state.record_success("update")  # freshness advanced → invalidate

    cache.get("bitcoin")
    assert len(calls) == 2


def test_bounded_size_default_128(tmp_path):
    embedder, calls = _spy_embedder()
    cache = ResultCache(embedder, tmp_path / "ingest.db")
    assert cache.size == 128
    for i in range(200):
        cache.get(f"query {i}")
    assert len(cache) == 128


def test_metrics_logged(tmp_path, caplog):
    import logging

    embedder, calls = _spy_embedder()
    cache = ResultCache(embedder, tmp_path / "ingest.db", size=4)
    with caplog.at_level(logging.DEBUG, logger="mc.search.cache"):
        cache.get("x")
        cache.get("x")
        cache.get("y")
        cache.get("x")
    assert cache.metrics == {"hits": 2, "misses": 2}
    assert any("hit" in r.message.lower() for r in caplog.records)


def test_engine_search_consults_cache(monkeypatch, tmp_path):
    """Review fix (Story 4.7 wiring): engine.search with a cache embeds the
    dense query once across identical queries — the crawler guard is live."""
    from mc.search import engine as engine_mod
    from mc.search.cache import ResultCache
    from mc.search.engine import search

    calls: list[str] = []
    real = engine_mod.embed_query

    def spy(query: str) -> list[float]:
        calls.append(query)
        return real(query)

    monkeypatch.setattr(engine_mod, "embed_query", spy)

    class StubClient:
        def query_points(self, **kwargs):
            return type("Resp", (), {"points": []})()

    cache = ResultCache(spy, tmp_path / "ingest.db", size=8)
    search(StubClient(), "bitcoin preço", collection="x", cache=cache)
    search(StubClient(), "bitcoin preço", collection="x", cache=cache)
    assert len(calls) == 1, "second identical query must be served from cache"
