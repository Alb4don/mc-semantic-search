"""Story 4.1 — query normalization & phrase (verbatim) detection."""

from __future__ import annotations

from mc.search.query import parse_query


def test_quoted_query_is_verbatim():
    q = parse_query('"bitcoin é bolha"')
    assert q.is_verbatim
    assert q.phrase == "bitcoin é bolha"


def test_unquoted_query_is_semantic():
    q = parse_query("mercado cripto")
    assert not q.is_verbatim
    assert q.phrase is None


def test_diacritic_folding_for_matching_only():
    q = parse_query("não")
    assert q.normalized == "nao"
    assert q.raw == "não", "display must keep the original text"


def test_casefold_and_whitespace():
    q = parse_query("  BITCOIN   É   Bolha  ")
    assert q.normalized == "bitcoin e bolha"


def test_no_boolean_operator_leakage():
    """Sparse encoders receive plain text — operators must be neutralized."""
    q = parse_query('bitcoin AND OR bolha "casa"')
    assert isinstance(q.normalized, str)
    assert "AND" not in q.normalized or "and" in q.normalized


def test_verbatim_phrase_normalized():
    q = parse_query('"MACRO é NÃO linear"')
    assert q.is_verbatim
    assert q.normalized_phrase == "macro e nao linear"
