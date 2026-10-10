"""BDD steps for tests/bdd/ui.feature (Story 4.8) — search page + htmx.

Parses the rendered HTML with the stdlib parser; assets are asserted
vendored (no external CDN requests, works offline and without JS).
"""

from __future__ import annotations

import re
import uuid

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.api.app import create_app
from mc.ingest.models import Chunk
from mc.qdrant import ensure_collection, upsert_video

scenarios("bdd/ui.feature")

pytestmark = pytest.mark.integration

QUERY_WORD = "bolha"


def _seed(ctx, query_word: str = QUERY_WORD) -> None:
    import mc.qdrant as q

    original = q.embed_passage
    q.embed_passage = lambda texts: [[0.5] * 384 for _ in texts]
    try:
        chunks = [
            Chunk(
                video_id="video000001",
                text=f"o mercado em crise e a pergunta se bitcoin é {query_word} volto já",
                url="https://www.youtube.com/watch?v=video000001&t=30",
                start_seconds=30,
                date="2026-10-09",
                episode="1245",
                channel="morning crypto",
            ),
            Chunk(
                video_id="video000002",
                text="a carteira fria guardou as chaves com segurança",
                url="https://www.youtube.com/watch?v=video000002&t=60",
                start_seconds=60,
                date="2026-10-08",
                episode="1244",
                channel="morning crypto",
            ),
        ]
        upsert_video(ctx["client"], ctx["collection"], "video000001", chunks[:1])
        upsert_video(ctx["client"], ctx["collection"], "video000002", chunks[1:])
        ctx["state"].record_success("update")
    finally:
        q.embed_passage = original


@pytest.fixture
def ctx(tmp_path):
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=30)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ensure_collection(client, collection)
    ctx = {"client": client, "collection": collection}
    from mc.state import StateStore

    ctx["state"] = StateStore(tmp_path / "ingest.db")
    _seed(ctx)
    ctx["app"] = create_app(
        client=client, collection=collection, state=ctx["state"]
    )
    ctx["client_http"] = TestClient(ctx["app"])
    ctx["empty_collection"] = None
    yield ctx
    client.delete_collection(collection)
    if ctx["empty_collection"]:
        client.delete_collection(ctx["empty_collection"])


@given("I am on the search page")
def open_page(ctx):
    ctx["response"] = ctx["client_http"].get("/")
    assert ctx["response"].status_code == 200
    ctx["html"] = ctx["response"].text


@when(parsers.parse('I type "{text}" in the search box'))
def type_query(ctx, text):
    # htmx live-swaps the /results partial as the user types
    ctx["partial"] = ctx["client_http"].get(
        "/results", params={"query": text}
    ).text
    # the no-JS fallback is a plain form GET rendering the full page
    ctx["response"] = ctx["client_http"].get("/", params={"query": text})
    ctx["html"] = ctx["response"].text


@then("results appear without a full page reload (htmx)")
def htmx_live_swap(ctx):
    assert "htmx.min.js" in ctx["html"]
    assert 'hx-get="/results"' in ctx["html"]
    assert 'hx-target="#resultados"' in ctx["html"]
    # the partial carries only the ledger — no <html> wrapper
    assert "<html" not in ctx["partial"]
    assert "tape-counter" in ctx["partial"]


@then("each result shows title, episode, date, channel and snippet")
def result_fields(ctx):
    for html in (ctx["html"], ctx["partial"]):
        assert 'class="hit-title"' in html
        assert "ep 1245" in html
        assert "2026-10-09" in html
        assert "morning crypto" in html
        assert f"<mark>{QUERY_WORD}</mark>" in html


@when(parsers.parse("I search for something with no match"))
def search_no_match(ctx):
    # a mismatch only happens on an empty archive — the dense branch of the
    # hybrid engine always returns best-effort ANN matches otherwise
    from pathlib import Path

    from mc.state import StateStore as SS

    empty_collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx["empty_collection"] = empty_collection
    ensure_collection(ctx["client"], empty_collection)
    app = create_app(
        client=ctx["client"],
        collection=empty_collection,
        state=SS(Path(ctx["state"].path)),
    )
    ctx["response"] = TestClient(app).get("/", params={"query": "zzz-nada"})
    ctx["html"] = ctx["response"].text


@then(
    parsers.parse(
        'the page shows "{headline}" with the archive freshness line'
    )
)
def empty_state(ctx, headline):
    assert headline in ctx["html"]
    assert "acervo atualizado até" in ctx["html"]


@then("the page does not look identical to a loading state")
def empty_vs_loading(ctx):
    # loading shows the htmx indicator; the empty card is a dashed panel
    assert 'class="empty-head"' in ctx["html"]
    assert "console-busy" in ctx["html"]


@given(parsers.parse("the archive was last updated {date}"))
def archive_updated(ctx, date):
    ctx["state"].record_key("last_success:update", f"{date}T06:15:00")


@when("I open the search page")
def open_page_again(ctx):
    ctx["response"] = ctx["client_http"].get("/")
    ctx["html"] = ctx["response"].text


@then(parsers.parse('the footer shows "acervo atualizado até {date}"'))
def footer_freshness(ctx, date):
    assert f"acervo atualizado até {date}" in ctx["html"]


def test_no_external_cdn_requests(ctx):
    """Vendored assets only — the page must render fully offline."""
    html = ctx["client_http"].get("/").text
    for match in re.findall(r'(?:src|href)="([^"]+)"', html):
        assert not match.startswith(("http://", "https://")), match
    for asset in (
        "/static/js/htmx.min.js",
        "/static/js/alpine.min.js",
        "/static/css/pico.min.css",
        "/static/css/mc.css",
    ):
        assert ctx["client_http"].get(asset).status_code == 200


def test_page_renders_without_javascript(ctx):
    """No-JS fallback: plain form GET renders the full results inline."""
    html = ctx["client_http"].get("/", params={"query": QUERY_WORD}).text
    assert "<html" in html and "tape-counter" in html
    assert 'action="/" method="get"' in html
