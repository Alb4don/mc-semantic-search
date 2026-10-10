"""BDD step definitions for tests/bdd/ranking.feature (Story 4.3).

Runs against the local qdrant container with the REAL embedders (e5
dense + fastembed BM25) and the REAL engine.search, on a throwaway
fixture collection (mc-bdd-<hex>) deleted on teardown. Ranking is
post-processed by mc.search.ranking.apply_boost driven by parse_query.
"""

from __future__ import annotations

import html
import re
import uuid

import pytest
from pytest_bdd import given, scenarios, then, when
from qdrant_client import QdrantClient

from mc.ingest.models import Chunk
from mc.qdrant import upsert_video
from mc.search.engine import search
from mc.search.query import normalize, parse_query
from mc.search.ranking import apply_boost
from mc.search.snippets import build_snippet

scenarios("bdd/ranking.feature")

pytestmark = pytest.mark.integration

LITERAL_PHRASE = "bitcoin é bolha"
LITERAL_VIDEO = "bddlit0001"
SEMANTIC_VIDEO = "bddsim0002"
INFLATION_VIDEOS = ("bddinfl0003", "bddinfl0004")

_DATE = "2026-10-01"


def _chunks() -> list[Chunk]:
    """Six realistic Portuguese chunks, one per video.

    v1 holds the verbatim phrase; v2 is semantically about it without the
    phrase; v3/v4 are about "mercado em alta mesmo com inflação"; the
    rest are unrelated fillers.
    """
    texts = {
        LITERAL_VIDEO: (
            "Tem gente repetindo por aí que bitcoin é bolha e que vai "
            "desabar tudo, mas eu olho para a oferta fixa e para a demanda "
            "institucional e enxergo exatamente o oposto."
        ),
        SEMANTIC_VIDEO: (
            "Depois de subir tanto neste ciclo, a maior criptomoeda do "
            "mundo chega a um preço esticado, e analistas comparam o "
            "movimento às bolhas clássicas dos mercados financeiros."
        ),
        INFLATION_VIDEOS[0]: (
            "Mesmo com a inflação corroendo o poder de compra, o mercado "
            "segue em alta: investidores trocam moeda fraca por ativos "
            "escassos e a bolsa reflete essa migração."
        ),
        INFLATION_VIDEOS[1]: (
            "O cenário é de mercado em alta mesmo com a inflação "
            "persistente, porque a expectativa de corte de juros empurra "
            "os ativos de risco para cima."
        ),
        "bddfill0005": (
            "A regulamentação de criptoativos avança no congresso com "
            "novas regras para corretoras e custódia de ativos digitais."
        ),
        "bddfill0006": (
            "Tokens de governança de protocolos de empréstimo "
            "descentralizado registraram forte valorização após a "
            "atualização da plataforma."
        ),
    }
    return [
        Chunk(
            video_id=vid,
            text=text,
            url=f"https://www.youtube.com/watch?v={vid}&t=0s",
            start_seconds=0,
            date=_DATE,
            episode="ep",
            channel="Morning Crypto",
        )
        for vid, text in texts.items()
    ]


@pytest.fixture
def ctx():
    try:
        client = QdrantClient(url="http://localhost:6333", timeout=120)
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    collection = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ctx = {"client": client, "collection": collection}
    corpus = _chunks()
    for vid in {c.video_id for c in corpus}:
        upsert_video(
            client, collection, vid, [c for c in corpus if c.video_id == vid]
        )
    yield ctx
    client.delete_collection(ctx["collection"])
    client.close()


def _text(vid: str) -> str:
    return next(c.text for c in _chunks() if c.video_id == vid)


# --- Given ---------------------------------------------------------------------
# NOTE: the corpus is indexed eagerly in the ctx fixture (scenario 2 has no
# Given steps); the Given steps below only assert the archive shape.


@given('the archive contains a chunk with the literal phrase "bitcoin é bolha"')
def archive_with_literal_chunk(ctx):
    assert normalize(LITERAL_PHRASE) in normalize(_text(LITERAL_VIDEO))


@given("a semantically similar chunk without that phrase")
def archive_with_similar_chunk(ctx):
    assert normalize(LITERAL_PHRASE) in normalize(_text(LITERAL_VIDEO))
    assert normalize(LITERAL_PHRASE) not in normalize(_text(SEMANTIC_VIDEO))


# --- When ----------------------------------------------------------------------
# Exact-match steps: parsers.parse would strip the quotes from the captured
# argument, silently turning the verbatim search into a paraphrase one.


@when('I search for "bitcoin é bolha"')
def run_search_verbatim(ctx):
    _run_search(ctx, f'"{LITERAL_PHRASE}"')


@when('I search for "mercado em alta mesmo com inflação"')
def run_search_paraphrase(ctx):
    _run_search(ctx, "mercado em alta mesmo com inflação")


def _run_search(ctx, raw_query: str) -> None:
    client = ctx["client"]
    ctx["raw_results"] = search(
        client, raw_query, collection=ctx["collection"], limit=10
    )
    ctx["query"] = parse_query(raw_query)
    ctx["results"] = apply_boost(ctx["raw_results"], ctx["query"])


# --- Then ----------------------------------------------------------------------


@then("the literal chunk is ranked first")
def literal_chunk_ranked_first(ctx):
    results = ctx["results"]
    assert results, "no results returned"
    assert results[0].video_id == LITERAL_VIDEO, [r.text[:60] for r in results]
    # boost actually demoted every non-literal result by the factor
    for raw, boosted in zip(ctx["raw_results"], results, strict=True):
        if boosted.video_id == LITERAL_VIDEO:
            assert boosted.score == pytest.approx(raw.score)  # literal untouched
        else:
            assert boosted.score == pytest.approx(raw.score * 0.5)


@then("its snippet contains the exact phrase")
def snippet_contains_exact_phrase(ctx):
    top = ctx["results"][0]
    snippet = build_snippet(top.text, ctx["query"].normalized)
    assert snippet.has_literal
    plain = html.unescape(re.sub(r"<[^>]+>", "", snippet.highlighted))
    assert normalize(LITERAL_PHRASE) in normalize(plain)


@then("the top results are chunks semantically about that idea")
def top_results_are_semantic_chunks(ctx):
    results = ctx["results"]
    ids = [r.video_id for r in results]
    assert set(INFLATION_VIDEOS) <= set(ids), ids
    assert set(INFLATION_VIDEOS) <= set(ids[:3]), ids[:3]


@then("results are demoted only by the query terms they lack")
def demoted_only_by_missing_terms(ctx):
    query = ctx["query"]
    assert not query.is_verbatim
    # Coverage rule: a chunk containing every content term keeps its raw
    # score; each missing term multiplies by the boost factor. Count and
    # texts are preserved — only scores shrink, by exact powers of 0.5.
    from mc.search.ranking import _content_terms

    terms = _content_terms(query.normalized)
    assert terms
    results = ctx["results"]
    assert len(results) == len(ctx["raw_results"])
    # apply_boost re-sorts, so pair with the raw scores by video id
    raw_by_id = {r.video_id: r for r in ctx["raw_results"]}
    for boosted in results:
        raw = raw_by_id[boosted.video_id]
        hay = normalize(raw.text or "")
        missing = sum(
            1
            for t in terms
            if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", hay) is None
        )
        assert boosted.score == pytest.approx(raw.score * 0.5**missing)
