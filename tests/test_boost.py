"""Story 4.3 — quote-aware ranking boost (pure unit tests, no Qdrant).

Property-style invariant (DoD): the boost NEVER changes the result count,
only the order/scores — verified deterministically over ~50 seeded
permutations with `random.Random(42)` (hypothesis is not installed).
"""

from __future__ import annotations

import random

import pytest

from mc import config
from mc.search.engine import SearchResult
from mc.search.query import normalize, parse_query
from mc.search.ranking import apply_boost

PHRASE = "bitcoin é bolha"
PHRASE_NORM = normalize(PHRASE)  # "bitcoin e bolha" (diacritic-folded)

FACTOR = 0.5


def _result(vid: str, score: float, *, literal: bool) -> SearchResult:
    if literal:
        text = "Todo mundo repete: bitcoin é bolha, mas eu vejo escassez e demanda."
    else:
        text = f"Hoje falamos da regulamentação de criptoativos e da custódia ({vid})."
    return SearchResult(
        video_id=vid,
        text=text,
        url=f"https://www.youtube.com/watch?v={vid}",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=score,
    )


def _is_literal(r: SearchResult) -> bool:
    return PHRASE_NORM in normalize(r.text)


VERBATIM = parse_query(f'"{PHRASE}"')
PLAIN = parse_query("regulamentação de criptoativos")


# --- Non-verbatim queries pass through untouched ------------------------------


def test_non_verbatim_returns_results_unchanged():
    results = [_result("a", 3.0, literal=False), _result("b", 1.0, literal=True)]
    boosted = apply_boost(results, PLAIN, factor=FACTOR)
    assert boosted == results
    assert [r.score for r in boosted] == [3.0, 1.0]


def test_non_verbatim_never_touches_scores_even_without_phrase():
    results = [_result("a", 2.0, literal=False)]
    assert apply_boost(results, PLAIN, factor=0.0) == results


# --- Verbatim: demote, never drop ---------------------------------------------


def test_verbatim_demotes_results_lacking_phrase():
    results = [
        _result("para", 2.0, literal=False),
        _result("lit", 1.0, literal=True),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert len(boosted) == 2  # never dropped
    by_id = {r.video_id: r for r in boosted}
    assert by_id["lit"].score == pytest.approx(1.0)  # literal untouched
    assert by_id["para"].score == pytest.approx(2.0 * FACTOR)


def test_verbatim_reorders_descending():
    results = [
        _result("para", 10.0, literal=False),
        _result("lit", 6.0, literal=True),
        _result("para2", 8.0, literal=False),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["lit", "para", "para2"]
    assert [r.score for r in boosted] == [6.0, 5.0, 4.0]


def test_verbatim_all_literal_keeps_scores_desc():
    results = [
        _result("lit1", 1.0, literal=True),
        _result("lit2", 5.0, literal=True),
    ]
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert [r.video_id for r in boosted] == ["lit2", "lit1"]
    assert [r.score for r in boosted] == [5.0, 1.0]


def test_count_never_changes_and_input_not_mutated():
    results = [
        _result("a", 3.0, literal=False),
        _result("b", 2.0, literal=True),
        _result("c", 1.0, literal=False),
    ]
    snapshot = list(results)
    boosted = apply_boost(results, VERBATIM, factor=FACTOR)
    assert len(boosted) == len(results) == 3
    assert results == snapshot  # input list untouched
    assert all(isinstance(r, SearchResult) for r in boosted)


def test_empty_list_stays_empty():
    assert apply_boost([], VERBATIM, factor=FACTOR) == []
    assert apply_boost([], PLAIN) == []


def test_result_without_text_does_not_crash():
    """Poisoned-index payload without `text` must demote, not raise (500)."""
    result = SearchResult(
        video_id="v1",
        text=None,
        url="https://www.youtube.com/watch?v=v1",
        start_seconds=0,
        date=None,
        episode=None,
        channel=None,
        score=1.0,
    )
    boosted = apply_boost([result], VERBATIM, factor=FACTOR)
    assert boosted[0].score == pytest.approx(FACTOR)  # demoted, not crashed


def test_factor_parameter_overrides_default():
    results = [_result("a", 10.0, literal=False)]
    (only,) = apply_boost(results, VERBATIM, factor=0.25)
    assert only.score == pytest.approx(2.5)


# --- Config default ------------------------------------------------------------


def test_factor_defaults_to_config_constant(monkeypatch):
    """`factor=None` reads config.SEARCH_QUOTE_BOOST at CALL time.

    Note: config.py reads the SEARCH_QUOTE_BOOST env var at import time
    into a module constant, so an env override can only take effect
    before interpreter start — the dynamic-read path is tested here by
    monkeypatching the module attribute instead.
    """
    monkeypatch.setattr(config, "SEARCH_QUOTE_BOOST", 0.3)
    results = [_result("a", 10.0, literal=False)]
    (only,) = apply_boost(results, VERBATIM)
    assert only.score == pytest.approx(3.0)


# --- Property test (DoD): count preserved, scores are a permutation -----------


def test_property_boost_preserves_count_and_permutes_scores():
    """Over 50 seeded permutations: same count, scores sorted desc, and every
    output score is exactly the original or the original * factor."""
    rng = random.Random(42)
    for _ in range(50):
        n = rng.randint(0, 12)
        results = [
            _result(f"v{i:02d}", rng.random() * 10 + 0.001, literal=rng.random() < 0.5)
            for i in range(n)
        ]
        boosted = apply_boost(results, VERBATIM, factor=FACTOR)

        assert len(boosted) == n  # count NEVER changes

        scores = [r.score for r in boosted]
        assert scores == sorted(scores, reverse=True)  # descending order

        # multiset of video ids preserved (pure permutation)
        assert sorted(r.video_id for r in boosted) == sorted(
            r.video_id for r in results
        )

        for out in boosted:
            orig = next(r for r in results if r.video_id == out.video_id)
            expected = orig.score if _is_literal(orig) else orig.score * FACTOR
            assert out.score == pytest.approx(expected)
            assert _is_literal(out) == _is_literal(orig)  # text untouched
