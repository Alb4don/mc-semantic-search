"""BDD step definitions for tests/bdd/api.feature (Story 4.5).

Runs against the local qdrant container with a throwaway fixture
collection. Passage embeddings are faked (constant 384-dim vectors) so
seeding is instant; the sparse encoder is the real BM25, so literal
keywords still drive ranking and every result can be snippet-checked.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.api.app import create_app
from mc.embeddings import embed_query
from mc.ingest.models import Chunk
from mc.qdrant import ensure_collection, upsert_video
from mc.state import StateStore

scenarios("bdd/api.feature")

pytestmark = pytest.mark.integration

CHUNK_COUNT = 55
VIDEO_IDS = ["video000001", "video000002", "video000003"]
QUERY_WORD = "bitcoin"

RESULT_KEYS = {
    "title",
    "url",
    "start_seconds",
    "date",
    "episode",
    "channel",
    "snippet",
    "score",
}


def _chunks() -> list[Chunk]:
    """55 single-sentence chunks, each containing the seeded query word."""
    return [
        Chunk(
            video_id=VIDEO_IDS[i % 3],
            text=f"o {QUERY_WORD} reagiu {i} vezes ao boato do dia.",
            url=f"https://www.youtube.com/watch?v={VIDEO_IDS[i % 3]}&t={(i // 3) * 180}s",
            start_seconds=(i // 3) * 180,
            date="2026-09-15",
            episode=f"{2000 + i}",
            channel="Morning Crypto",
        )
        for i in range(CHUNK_COUNT)
    ]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=30)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx = {
        "client": client,
        "collection": collection,
        "response": None,
        "app": None,
    }
    yield ctx
    client.delete_collection(collection)


def _make_app(ctx, state: StateStore):
    return create_app(
        client=ctx["client"],
        collection=ctx["collection"],
        state=state,
    )


@given(
    parsers.parse("a seeded archive of {n:d} transcript chunks"),
    target_fixture="state",
)
def seeded_archive(ctx, tmp_path, monkeypatch, n):
    # All passage vectors equal the real e5 embedding of the query word, so
    # dense similarity is uniformly positive and the zero-vector meta sentinel
    # ranks LAST (a constant vector would make the meta point rank first).
    seed_vector = embed_query(QUERY_WORD)
    monkeypatch.setattr(
        "mc.qdrant.embed_passage",
        lambda texts: [list(seed_vector) for _ in texts],
    )
    corpus = _chunks()[:n]
    for vid in dict.fromkeys(VIDEO_IDS):
        batch = [c for c in corpus if c.video_id == vid]
        if batch:
            upsert_video(ctx["client"], ctx["collection"], vid, batch)
    state = StateStore(tmp_path / "ingest.db")
    state.record_success("update")
    ctx["app"] = _make_app(ctx, state)
    return state


@given("an empty collection", target_fixture="state")
def empty_collection(ctx, tmp_path):
    ensure_collection(ctx["client"], ctx["collection"])
    state = StateStore(tmp_path / "ingest.db")
    ctx["app"] = _make_app(ctx, state)
    return state


@given("the search service has no reachable qdrant")
def unreachable_qdrant(ctx, tmp_path):
    dead = QdrantClient(url="http://localhost:1", timeout=0.2)
    state = StateStore(tmp_path / "ingest.db")
    ctx["app"] = create_app(client=dead, collection="mc-bdd-nonexistent", state=state)


@when(parsers.parse('I search for "{query}"'))
def search(ctx, query):
    ctx["response"] = TestClient(ctx["app"]).get(
        "/search", params={"query": query}
    )


@when(parsers.parse('I search with query "{query}" and limit {limit:d}'))
def search_with_limit(ctx, query, limit):
    ctx["response"] = TestClient(ctx["app"]).get(
        "/search", params={"query": query, "limit": limit}
    )


@then(parsers.parse("the response is {code:d}"))
def response_status(ctx, code):
    assert ctx["response"].status_code == code, ctx["response"].text


@then("the body matches the /search contract")
def contract_shape(ctx):
    body = ctx["response"].json()
    assert set(body) == {"query", "took_ms", "freshness", "results"}
    assert body["query"] == "bitcoin"
    assert isinstance(body["took_ms"], int | float)
    assert "last_update" in body["freshness"]
    assert isinstance(body["results"], list)
    for result in body["results"]:
        assert set(result) == RESULT_KEYS
        assert result["url"].startswith("https://www.youtube.com/watch?v=")
        assert isinstance(result["start_seconds"], int)
        assert isinstance(result["score"], float)


@then("every result snippet highlights the query word")
def snippets_highlighted(ctx):
    for result in ctx["response"].json()["results"]:
        assert f"<mark>{QUERY_WORD}</mark>" in result["snippet"], result["snippet"]


@then(parsers.parse("exactly {n:d} results are returned"))
def exactly_n_results(ctx, n):
    assert len(ctx["response"].json()["results"]) == n


@then("the results list is empty")
def empty_results(ctx):
    assert ctx["response"].json()["results"] == []


@then("the freshness block is present")
def freshness_block(ctx):
    body = ctx["response"].json()
    assert "freshness" in body and "last_update" in body["freshness"]


def test_verbatim_query_applies_quote_boost_via_api(tmp_path, monkeypatch):
    """Story 4.3 integration: /search demotes non-literal hits on verbatim queries."""
    from mc.api import app as app_module
    from mc.search.engine import SearchResult
    from mc.state import StateStore

    literal = SearchResult(
        video_id="video000001",
        text="o mercado em crise, bitcoin é bolha disse ele",
        url="https://www.youtube.com/watch?v=video000001&t=30",
        start_seconds=30,
        date="2026-01-01",
        episode="1",
        channel="mc",
        score=1.0,
    )
    paraphrase = SearchResult(
        video_id="video000002",
        text="o mercado de cripto está muito caro e exagerado",
        url="https://www.youtube.com/watch?v=video000002&t=60",
        start_seconds=60,
        date="2026-01-01",
        episode="2",
        channel="mc",
        score=1.5,  # higher raw score, but lacks the literal phrase
    )

    def fake_search(client, query_raw, *, limit=10, collection=None, cache=None):
        return [paraphrase, literal]

    monkeypatch.setattr(app_module, "run_search", fake_search)
    state = StateStore(tmp_path / "ingest.db")
    client = TestClient(app_module.create_app(client=object(), state=state))
    body = client.get("/search", params={"query": '"bitcoin é bolha"'}).json()

    scores = [r["score"] for r in body["results"]]
    assert body["results"][0]["episode"] == "1"  # literal chunk first
    assert len(body["results"]) == 2  # never dropped
    assert scores[1] == 0.75  # 1.5 * factor 0.5 — paraphrase demoted
