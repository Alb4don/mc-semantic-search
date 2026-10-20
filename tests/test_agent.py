"""Agent-facing surface (unit, no qdrant): render=json envelope, the
no-query manual, /llms.txt, and the instructions section on the landing
page. run_search is faked like test_api_search.py does."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from mc.api import app as app_module
from mc.api.app import create_app
from mc.search.engine import SearchResult
from mc.state import StateStore

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


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        app_module,
        "run_search",
        lambda *args, **kwargs: [
            SearchResult(
                video_id="video000001",
                text="o bitcoin reagiu ao boato do dia",
                url="https://www.youtube.com/watch?v=video000001&t=30",
                start_seconds=30,
                date="2026-01-01",
                episode="1035",
                channel="mc",
                score=1.0,
            )
        ],
    )
    state = StateStore(tmp_path / "ingest.db")
    return TestClient(create_app(client=object(), state=state))


def test_render_json_with_query_matches_search_contract(client):
    response = client.get("/", params={"query": "bitcoin", "render": "json"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert set(body) == {"agent", "query", "took_ms", "freshness", "results"}
    assert body["query"] == "bitcoin"
    assert isinstance(body["took_ms"], int | float)
    assert "last_update" in body["freshness"]
    assert len(body["results"]) == 1
    assert set(body["results"][0]) == RESULT_KEYS
    assert body["results"][0]["url"].startswith("https://www.youtube.com/watch?v=")


def test_render_json_envelope_is_self_describing(client):
    """Every JSON answer carries Muriel's card so the agent can chain calls."""
    agent = client.get("/", params={"query": "bitcoin", "render": "json"}).json()[
        "agent"
    ]
    assert agent["name"] == "Muriel"
    assert any("/?query=SEU_TERMO&render=json" in line for line in agent["how_to_search"])
    assert any("/search?query=SEU_TERMO&limit=10" in line for line in agent["how_to_search"])
    assert set(agent["result_fields"]) == RESULT_KEYS


def test_render_json_without_query_returns_manual(client):
    response = client.get("/", params={"render": "json"})
    body = response.json()
    assert body["query"] == ""
    assert body["results"] == []
    assert body["agent"]["role"] == "bibliotecária do acervo Morning Crypto"


def test_render_is_case_insensitive_and_defaults_to_html(client):
    json_body = client.get("/", params={"query": "bitcoin", "render": "JSON"})
    assert json_body.headers["content-type"].startswith("application/json")
    html_body = client.get("/", params={"query": "bitcoin", "render": "xml"})
    assert html_body.headers["content-type"].startswith("text/html")
    default_body = client.get("/", params={"query": "bitcoin"})
    assert default_body.headers["content-type"].startswith("text/html")


def test_oversized_render_param_rejected(client):
    assert client.get("/", params={"render": "a" * 17}).status_code == 422
    assert client.get("/", params={"render": "a" * 16}).status_code == 200


def test_llms_txt_document(client):
    response = client.get("/llms.txt")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    assert text.startswith("# Muriel")
    # canonical link-list items: sections only accept `- [name](url): notes`
    # (strict validators reject plain text lines inside H2 sections)
    assert (
        "- [Busca render=json](http://testserver/?query=SEU_TERMO&render=json):"
        in text
    )
    assert (
        "- [API /search](http://testserver/search?query=SEU_TERMO&limit=10):"
        in text
    )
    assert "start_seconds" in text  # result fields are documented
    assert "<html" not in text  # plain text, never the page


def test_llms_txt_sections_are_canonical_link_lists(client):
    """Regression for llmstxtvalidator.org: inside an H2 section the only
    allowed lines are `- [name](url): optional notes`; anything else
    (GET recipes, bare field docs) fails the validator as plain text."""
    in_section = False
    for line in client.get("/llms.txt").text.splitlines():
        if line.startswith("## "):
            in_section = True
        elif in_section and line.strip():
            assert re.match(r"^- \[[^\]]+\]\([^)\s]+\)(?:: .+)?$", line), line


def test_llms_txt_honors_proxy_scheme(client):
    response = client.get("/llms.txt", headers={"x-forwarded-proto": "https"})
    assert "https://testserver/llms.txt" in response.text


def test_landing_page_carries_agent_instructions(client):
    html = client.get("/").text
    assert 'class="agent-note"' in html
    assert "Fale com a Muriel" in html
    # the paste-ready phrase points the agent at the machine-readable manual
    assert "quando o eddie falou sobre ilha das américas?" in html
    assert "https://muriel.eddieoz.com/llms.txt" in html
    assert 'href="/llms.txt"' in html
    # the human surface carries no endpoint jargon — agents get the full
    # recipe in /llms.txt and the render=json envelope instead
    assert "SEU_TERMO" not in html
    assert "GET /search" not in html
    # the box survives a query too — agents land on result URLs
    assert "agent-note" in client.get("/", params={"query": "bitcoin"}).text
