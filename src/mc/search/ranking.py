"""Literal-aware ranking boost (Story 4.3; revised for natural queries).

Deterministic post-processing of engine results. Two rules, both
order-only — the result count NEVER changes, recall is preserved:

- Verbatim (quoted phrase): results whose normalized text lacks the
  phrase are DEMOTED — score multiplied by SEARCH_QUOTE_BOOST.
- Natural language (unquoted): results are demoted by
  SEARCH_QUOTE_BOOST ** (number of query content words missing from the
  chunk). A chunk containing every content word keeps its score; each
  missing word halves it. Words shorter than 3 characters are ignored as
  content terms — after diacritic folding the copula "é" becomes the
  conjunction "e", which appears in nearly every Portuguese sentence and
  carries no signal (this also mirrors snippets.MIN_HIGHLIGHT_LEN).

Results are re-sorted by score descending (stable: ties keep their
original relative order). If no result matches any term, every score is
scaled by the same factor and the order is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import replace

from mc import config
from mc.search.engine import SearchResult
from mc.search.query import Query, normalize

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Content-term floor, mirroring snippets.MIN_HIGHLIGHT_LEN: 1-2 char tokens
# ("e", "é", "de", "a") are function words with no discrimination power.
MIN_TERM_LEN = 3


def _content_terms(normalized_query: str) -> list[str]:
    """Distinct folded query words long enough to carry signal."""
    terms: list[str] = []
    for word in _WORD_RE.findall(normalized_query):
        if len(word) >= MIN_TERM_LEN and word not in terms:
            terms.append(word)
    return terms


def _coverage_missing(haystack: str, patterns: list[re.Pattern[str]]) -> int:
    """How many term patterns are absent from the (normalized) chunk text."""
    return sum(1 for pattern in patterns if not pattern.search(haystack))


def apply_boost(
    results: list[SearchResult],
    query: Query,
    factor: float | None = None,
) -> list[SearchResult]:
    """Demote non-literal results; count never changes.

    `factor=None` reads `config.SEARCH_QUOTE_BOOST` at CALL time so an env
    override (or a test monkeypatching the module attribute) is honored
    without re-importing anything.
    """
    if factor is None:
        factor = config.SEARCH_QUOTE_BOOST
    if not results:
        return []

    if query.is_verbatim:
        phrase = query.normalized_phrase
        if not phrase:
            return list(results)
        scaled = [
            r if phrase in normalize(r.text or "") else replace(r, score=r.score * factor)
            for r in results
        ]
    else:
        terms = _content_terms(query.normalized)
        if not terms or len(results) <= 1:
            return list(results)
        patterns = [
            re.compile(rf"(?<!\w){re.escape(term)}(?!\w)") for term in terms
        ]
        scaled = []
        for r in results:
            missing = _coverage_missing(normalize(r.text or ""), patterns)
            scaled.append(
                r if not missing else replace(r, score=r.score * factor**missing)
            )

    # Stable sort: ties keep their original relative order.
    scaled.sort(key=lambda r: r.score, reverse=True)
    return scaled
