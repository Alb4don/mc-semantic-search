"""Story 4.2 — hybrid search engine with native Qdrant RRF fusion.

Integration tests against local Qdrant with a throwaway fixture collection
of ~10 realistic Portuguese chunks, using the REAL embedders (e5 + BM25).
"""

from __future__ import annotations

import uuid

import pytest
from qdrant_client import QdrantClient

import mc.search.engine as engine_mod
from mc.ingest.models import Chunk
from mc.qdrant import ensure_collection, upsert_video
from mc.search.engine import SearchResult, search

pytestmark = pytest.mark.integration

# Distinctive literal keyword phrase — sparse/BM25 must pull its chunk up.
KEYWORD_MARKER = "custo médio de mineração"
KEYWORD_QUERY = "custo médio de mineração do bitcoin"

# Semantically-close chunk described with completely different wording —
# dense/e5 must pull it up (no lexical overlap with the query).
SEMANTIC_MARKER = "lastro suficiente"
SEMANTIC_QUERY = (
    "reservas em dólar das moedas ancoradas não cobrem o volume emitido"
)

_DATE = "2026-09-15"


def _corpus() -> list[Chunk]:
    """Ten realistic Portuguese crypto chunks spread over three videos."""
    texts = [
        # 0 — keyword target: literal phrase "custo médio de mineração".
        (
            "Hoje o custo médio de mineração do bitcoin passou de quarenta mil "
            "dólares por moeda, e a dificuldade da rede ajustou para cima mais "
            "uma vez."
        ),
        # 1
        (
            "O mercado de ETFs à vista de ethereum recebeu um volume recorde de "
            "aplicações nesta semana, impulsionado por fundos institucionais."
        ),
        # 2 — semantic target: dollar-pegged assets issued without backing.
        (
            "A emissora de ativos atrelados ao dólar não mantém lastro "
            "suficiente para cobrir tudo que circulou no mercado, e o "
            "descasamento entre oferta e reservas assusta os investidores."
        ),
        # 3
        (
            "O halving reduziu a emissão diária de novas moedas, e a escassez "
            "programada volta a alimentar projeções otimistas para o ciclo."
        ),
        # 4
        (
            "A regulamentação de criptoativos avança no congresso com novas "
            "regras para corretoras e custódia de ativos digitais."
        ),
        # 5
        (
            "Tokens de governança de protocolos de empréstimo descentralizado "
            "registraram forte valorização após a atualização da plataforma."
        ),
        # 6
        (
            "A taxa de hash da rede alcançou um novo máximo histórico, "
            "refletindo a chegada de nova infraestrutura de mineração ao "
            "parque industrial."
        ),
        # 7
        (
            "Carteiras institucionais acumulam moedas ancoradas à espera de um "
            "momento mais favorável para entrar no mercado à vista."
        ),
        # 8
        (
            "A corrida pelos fundos de cripto impulsiona o volume negociado "
            "nas bolsas americanas, com entradas líquidas por vários dias."
        ),
        # 9
        (
            "Projetos de tokenização de ativos reais ganham força entre bancos, "
            "que testam contratos inteligentes para escrituração de títulos."
        ),
    ]
    videos = ["video000001", "video000002", "video000003"]
    chunks: list[Chunk] = []
    for i, text in enumerate(texts):
        vid = videos[i % 3]
        start = (i // 3) * 180
        chunks.append(
            Chunk(
                video_id=vid,
                text=text,
                url=f"https://www.youtube.com/watch?v={vid}&t={start}s",
                start_seconds=start,
                date=_DATE,
                episode=f"{1000 + i}",
                channel="Morning Crypto",
            )
        )
    return chunks


@pytest.fixture
def client() -> QdrantClient:
    client = QdrantClient(url="http://localhost:6333", timeout=120)
    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"local qdrant unreachable: {exc}")
    return client


@pytest.fixture
def coll(client: QdrantClient) -> str:
    name = f"mc-bdd-{uuid.uuid4().hex[:8]}"
    ensure_collection(client, name)
    corpus = _corpus()
    for vid in {c.video_id for c in corpus}:
        upsert_video(client, name, vid, [c for c in corpus if c.video_id == vid])
    yield name
    client.delete_collection(name)


def test_keyword_exact_query_ranks_literal_chunk_top3(client, coll):
    results = search(client, KEYWORD_QUERY, collection=coll, limit=10)
    assert len(results) == 10
    assert any(KEYWORD_MARKER in r.text for r in results[:3]), [
        r.text[:60] for r in results[:3]
    ]


def test_paraphrase_query_ranks_semantic_chunk_top3(client, coll):
    results = search(client, SEMANTIC_QUERY, collection=coll, limit=10)
    assert len(results) == 10
    assert any(SEMANTIC_MARKER in r.text for r in results[:3]), [
        r.text[:60] for r in results[:3]
    ]


def test_results_are_typed_with_payload_fields(client, coll):
    results = search(client, "regulamentação de criptoativos", collection=coll, limit=5)
    assert results and all(isinstance(r, SearchResult) for r in results)
    for r in results:
        assert r.video_id.startswith("video")
        assert r.text
        assert r.url.startswith("https://www.youtube.com/watch?v=")
        assert "&t=" in r.url
        assert r.date == _DATE
        assert r.episode
        assert r.channel == "Morning Crypto"
        assert isinstance(r.start_seconds, int)
        assert isinstance(r.score, float) and r.score > 0


def test_embeddings_run_in_single_worker_executor(client, coll, monkeypatch):
    """NFR-1: both embedder calls go through a ThreadPoolExecutor(max_workers=1)."""
    real_executor = engine_mod.ThreadPoolExecutor
    observed: dict = {"max_workers": None, "submits": 0}

    class SpyExecutor(real_executor):
        def __init__(self, *args, max_workers=None, **kwargs):
            observed["max_workers"] = max_workers
            super().__init__(*args, max_workers=max_workers, **kwargs)

        def submit(self, *args, **kwargs):
            observed["submits"] += 1
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(engine_mod, "ThreadPoolExecutor", SpyExecutor)
    search(client, "halving", collection=coll, limit=3)
    assert observed["max_workers"] == 1
    assert observed["submits"] == 2


def test_tokenless_query_returns_results_gracefully(client, coll):
    """A query whose sparse vector has no non-zero terms must not explode."""
    results = search(client, "!!!", collection=coll, limit=5)
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
