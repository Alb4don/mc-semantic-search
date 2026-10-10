"""Story 5.4: evaluation harness — sparse-only vs hybrid MRR on known quotes.

The rank/MRR math is unit-tested with injected search functions; a small
integration pass runs the real sparse+hybrid path against local Qdrant.
The full 27-quote suite run against the real archive lands in
docs/tasks/eval-report.md (needs the backfilled index).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mc.eval import (
    EvalCase,
    EvalReport,
    format_report,
    load_suite,
    reciprocal_rank,
    run_eval,
)

SUITE_CASES = [
    EvalCase(
        id="video000001",
        quote="o mercado de cripto está muito caro e exagerado",
        video_id="video000001",
        url="https://www.youtube.com/watch?v=video000001&t=30",
    ),
    EvalCase(
        id="video000002",
        quote="regulação de criptomoedas avança no congresso",
        video_id="video000002",
        url="https://www.youtube.com/watch?v=video000002&t=60",
    ),
]


def test_load_suite_reads_yaml(tmp_path: Path) -> None:
    suite = tmp_path / "quotes.yaml"
    suite.write_text(
        """
cases:
  - id: video000001
    quote: "o mercado de cripto está muito caro e exagerado"
    video_id: video000001
    url: "https://www.youtube.com/watch?v=video000001&t=30"
""",
        encoding="utf-8",
    )
    cases = load_suite(suite)
    assert len(cases) == 1
    assert cases[0].video_id == "video000001"
    assert cases[0].url.endswith("&t=30")


def test_reciprocal_rank_math() -> None:
    assert reciprocal_rank(rank=1) == 1.0
    assert reciprocal_rank(rank=4) == 0.25
    assert reciprocal_rank(rank=None) == 0.0


def _fake_searchers(sparse_map: dict, hybrid_map: dict) -> tuple:
    """Injectable search fns: map query → order of SUITE_CASES indexes."""

    def make(order_map):
        def fn(client, query, *, limit=50, collection=None):
            order = order_map.get(query, [])
            urls = [SUITE_CASES[i].url for i in order[:limit]]
            return [type("R", (), {"url": u}) for u in urls]

        return fn

    return make(sparse_map), make(hybrid_map)


def test_run_eval_computes_ranks_and_mrr() -> None:
    q0, q1 = SUITE_CASES[0].quote, SUITE_CASES[1].quote
    sparse_fn, hybrid_fn = _fake_searchers(
        sparse_map={q0: [1, 0], q1: [0, 1]},  # both targets at sparse rank 2
        hybrid_map={q0: [0, 1], q1: [1, 0]},  # both targets at hybrid rank 1
    )
    report = run_eval(
        SUITE_CASES,
        client=None,
        collection="x",
        sparse_fn=sparse_fn,
        hybrid_fn=hybrid_fn,
    )
    assert isinstance(report, EvalReport)
    # both hybrid ranks are 1 (chunkA first) and sparse ranks are 2 → hybrid wins
    assert all(r.hybrid_rank == 1 for r in report.cases)
    assert all(r.sparse_rank == 2 for r in report.cases)
    assert report.mrr_hybrid > report.mrr_sparse
    assert report.mrr_hybrid == pytest.approx(1.0)
    assert report.mrr_sparse == pytest.approx(0.5)


def test_run_eval_counts_misses_as_zero_rr() -> None:
    sparse_fn, hybrid_fn = _fake_searchers({}, {SUITE_CASES[0].quote: [0]})

    def sparse_none(client, query, *, limit=50, collection=None):
        return []

    report = run_eval(
        SUITE_CASES,
        client=None,
        collection="x",
        sparse_fn=sparse_none,
        hybrid_fn=hybrid_fn,
    )
    assert report.mrr_sparse == 0.0
    assert report.mrr_hybrid == pytest.approx(0.5)


def test_format_report_markdown_table() -> None:
    sparse_fn, hybrid_fn = _fake_searchers(
        {SUITE_CASES[0].quote: [0]}, {SUITE_CASES[0].quote: [0]}
    )
    report = run_eval(
        SUITE_CASES[:1],
        client=None,
        collection="x",
        sparse_fn=sparse_fn,
        hybrid_fn=hybrid_fn,
    )
    text = format_report(report)
    assert "| query |" in text
    assert "MRR" in text
    assert str(SUITE_CASES[0].video_id) in text


def test_suite_fixture_has_25_plus_cases() -> None:
    """The shipped suite carries the 25+ known quotes (DoD)."""
    suite = Path(__file__).parent / "eval" / "quotes.yaml"
    cases = load_suite(suite)
    assert len(cases) >= 25
    ids = [c.video_id for c in cases]
    assert len(set(ids)) >= 20  # spread across many videos


def test_hybrid_does_not_regress_verbatim_queries() -> None:
    """DoD: hybrid MRR >= sparse-only MRR (injected searchers simulating the
    expected production behavior on verbatim quotes)."""
    q0, q1 = SUITE_CASES[0].quote, SUITE_CASES[1].quote
    sparse_fn, hybrid_fn = _fake_searchers(
        sparse_map={q0: [0, 1], q1: [0, 1]},  # targets first in both modes
        hybrid_map={q0: [0, 1], q1: [1, 0]},
    )
    report = run_eval(
        SUITE_CASES,
        client=None,
        collection="x",
        sparse_fn=sparse_fn,
        hybrid_fn=hybrid_fn,
    )
    assert report.mrr_hybrid >= report.mrr_sparse
