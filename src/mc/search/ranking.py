"""Quote-aware ranking boost (Story 4.3).

Pure, deterministic post-processing of engine results: when the query is
verbatim (a quoted phrase), every result whose normalized text lacks the
phrase is DEMOTED — score multiplied by a factor — but never dropped, so
the result count can only stay the same and recall is preserved. Results
are re-sorted by the new score descending; literal matches and all
non-verbatim queries pass through unchanged.
"""

from __future__ import annotations

from dataclasses import replace

from mc import config
from mc.search.engine import SearchResult
from mc.search.query import Query, normalize


def apply_boost(
    results: list[SearchResult],
    query: Query,
    factor: float | None = None,
) -> list[SearchResult]:
    """Demote non-literal results on verbatim queries; count never changes.

    `factor=None` reads `config.SEARCH_QUOTE_BOOST` at CALL time so an env
    override (or a test monkeypatching the module attribute) is honored
    without re-importing anything.
    """
    if factor is None:
        factor = config.SEARCH_QUOTE_BOOST

    if not query.is_verbatim:
        return list(results)

    phrase = query.normalized_phrase
    demoted = [
        r if phrase and phrase in normalize(r.text or "") else replace(r, score=r.score * factor)
        for r in results
    ]
    # Stable sort: ties keep their original relative order.
    demoted.sort(key=lambda r: r.score, reverse=True)
    return demoted
