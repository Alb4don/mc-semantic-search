"""Evaluation harness (Story 5.4): sparse-only vs hybrid on known quotes.

Compares a sparse-only (BM25) search against the full hybrid pipeline
(RRF fusion + quote-aware boost) over a suite of known Portuguese
quotes, scoring each mode by the reciprocal rank of the literal chunk
that is known to contain the quote. Metric: mean reciprocal rank (MRR).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from qdrant_client import QdrantClient, models

from mc.config import COLLECTION_NAME, QDRANT_URL
from mc.embeddings import embed_sparse
from mc.qdrant import client_kwargs
from mc.search.engine import SearchResult
from mc.search.engine import search as hybrid_search
from mc.search.query import parse_query
from mc.search.ranking import apply_boost

SearchFn = Callable[..., list[SearchResult]]


@dataclass(frozen=True)
class EvalCase:
    id: str
    quote: str
    video_id: str
    url: str


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    sparse_rank: int | None
    hybrid_rank: int | None

    @property
    def sparse_rr(self) -> float:
        return reciprocal_rank(self.sparse_rank)

    @property
    def hybrid_rr(self) -> float:
        return reciprocal_rank(self.hybrid_rank)


@dataclass(frozen=True)
class EvalReport:
    cases: list[CaseResult]
    mrr_sparse: float
    mrr_hybrid: float
    took_seconds: float


def load_suite(path: str | Path) -> list[EvalCase]:
    """Load the quote suite YAML ({cases: [{id, quote, video_id, url}]})."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=str(c["id"]),
            quote=str(c["quote"]),
            video_id=str(c["video_id"]),
            url=str(c["url"]),
        )
        for c in data["cases"]
    ]


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def _find_rank(results: list[SearchResult], case: EvalCase) -> int | None:
    """1-based rank of the case's literal chunk in the result list.

    Exact chunk-URL match only (review fix): counting the first result of
    the case's *video* inflated MRR and could flip the regression verdict.
    """
    return next(
        (i for i, r in enumerate(results, start=1) if r.url == case.url),
        None,
    )


def sparse_only_search(
    client: QdrantClient, query_raw: str, *, limit: int = 50, collection: str
) -> list[SearchResult]:
    """Plain BM25 ANN search (single sparse vector, no fusion, no boost)."""
    parsed = parse_query(query_raw)
    vec = embed_sparse([parsed.normalized])[0]
    if not vec.values:
        return []
    response = client.query_points(
        collection_name=collection,
        using="sparse",
        query=models.SparseVector(indices=list(vec.indices), values=list(vec.values)),
        limit=limit,
        with_payload=True,
    )
    out: list[SearchResult] = []
    for point in response.points:
        payload = point.payload or {}
        if payload.get("__meta__"):
            continue
        out.append(
            SearchResult(
                video_id=payload.get("video_id", ""),
                text=payload.get("text", ""),
                url=payload.get("url", ""),
                start_seconds=payload.get("start_seconds", 0),
                date=payload.get("date"),
                episode=payload.get("episode"),
                channel=payload.get("channel"),
                score=float(point.score or 0.0),
            )
        )
    return out


def run_eval(
    cases: list[EvalCase],
    *,
    client: QdrantClient | None = None,
    collection: str = COLLECTION_NAME,
    limit: int = 50,
    sparse_fn: SearchFn | None = None,
    hybrid_fn: SearchFn | None = None,
) -> EvalReport:
    """Score both modes per case; MRR over reciprocal ranks (misses = 0)."""
    client = client or QdrantClient(url=QDRANT_URL, **client_kwargs(timeout=60))
    sparse_fn = sparse_fn or (
        lambda cl, q, **kw: sparse_only_search(cl, q, collection=collection, **kw)
    )
    hybrid_fn = hybrid_fn or (
        lambda cl, q, **kw: _hybrid_with_boost(cl, q, collection=collection, **kw)
    )
    started = time.perf_counter()
    results: list[CaseResult] = []
    for case in cases:
        sparse_rank = _find_rank(list(sparse_fn(client, case.quote, limit=limit)), case)
        hybrid_rank = _find_rank(list(hybrid_fn(client, case.quote, limit=limit)), case)
        results.append(
            CaseResult(
                case=case, sparse_rank=sparse_rank, hybrid_rank=hybrid_rank
            )
        )
    n = max(len(results), 1)
    return EvalReport(
        cases=results,
        mrr_sparse=sum(r.sparse_rr for r in results) / n,
        mrr_hybrid=sum(r.hybrid_rr for r in results) / n,
        took_seconds=time.perf_counter() - started,
    )


def _hybrid_with_boost(
    client: QdrantClient, query_raw: str, *, limit: int = 50, collection: str
) -> list[SearchResult]:
    query = parse_query(query_raw)
    results = hybrid_search(client, query_raw, limit=limit, collection=collection)
    return apply_boost(results, query)


def format_report(report: EvalReport) -> str:
    """Markdown: verdict line, per-query table, regression check."""
    lines = [
        "# Evaluation report — sparse-only vs hybrid (Story 5.4)",
        "",
        f"- MRR sparse-only: **{report.mrr_sparse:.3f}**",
        f"- MRR hybrid (RRF + quote boost): **{report.mrr_hybrid:.3f}**",
        f"- Cases: {len(report.cases)} — took {report.took_seconds:.1f}s",
        (
            "- Regression check: hybrid "
            + ("DOES NOT regress" if report.mrr_hybrid >= report.mrr_sparse else "REGRESSES")
            + " vs sparse-only on verbatim quotes."
        ),
        "",
        "| query | video | sparse rank | hybrid rank |",
        "|---|---|---|---|",
    ]
    for r in report.cases:
        quote = r.case.quote[:60].replace("|", "\\|")
        lines.append(
            f"| {quote} | {r.case.video_id} | {r.sparse_rank or '—'} "
            f"| {r.hybrid_rank or '—'} |"
        )
    return "\n".join(lines) + "\n"
