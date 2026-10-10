"""Story 4.9 tests: embedded player at the exact moment.

Privacy + behavior contract of the click-to-load player: youtube-nocookie
only, start=<chunk offset>, lazy-loaded, and not rendered until the user
opens the <details> placeholder.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from pytest_bdd import given, parsers, scenarios, then, when
from qdrant_client import QdrantClient

from mc.api.app import create_app
from mc.ingest.models import Chunk
from mc.qdrant import ensure_collection, upsert_video
from mc.state import StateStore

scenarios("bdd/player.feature")

pytestmark = pytest.mark.integration

CHUNK_OFFSET = 1534


@pytest.fixture
def page(tmp_path):
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=30)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ensure_collection(client, collection)
    import mc.qdrant as q

    original = q.embed_passage
    q.embed_passage = lambda texts: [[0.5] * 384 for _ in texts]
    try:
        upsert_video(
            client,
            collection,
            "video000001",
            [
                Chunk(
                    video_id="video000001",
                    text="a auditoria do contrato começou agora com atenção total",
                    url=f"https://www.youtube.com/watch?v=video000001&t={CHUNK_OFFSET}",
                    start_seconds=CHUNK_OFFSET,
                    date="2026-10-09",
                    episode="1534",
                    channel="morning crypto",
                )
            ],
        )
    finally:
        q.embed_passage = original
    state = StateStore(tmp_path / "ingest.db")
    http = TestClient(create_app(client=client, collection=collection, state=state))
    html = http.get("/", params={"query": "auditoria do contrato"}).text
    yield html
    client.delete_collection(collection)


@given(parsers.parse("a result in the archive with a chunk at {offset:d} seconds"))
def result_with_chunk(page):
    assert page


@when("I open the result's player placeholder")
def open_player(page):
    assert "<details" in page  # placeholder exists in the markup


@then(parsers.parse("the iframe points at youtube-nocookie with start={offset:d}"))
def iframe_nocookie_start(page, offset):
    assert f"https://www.youtube-nocookie.com/embed/video000001?start={offset}" in page


@then("the iframe is lazy-loaded")
def iframe_lazy(page):
    assert 'loading="lazy"' in page


@then("the player is not rendered until user action (click-to-load)")
def player_click_to_load(page):
    # the iframe lives inside <details>, so browsers fetch it only on open
    details_open = page.index("<details")
    iframe_at = page.index("<iframe")
    assert details_open < iframe_at
    # privacy: no tracking youtube.com embed anywhere
    assert "youtube.com/embed" not in page
    assert "youtube-nocookie.com/embed" in page
