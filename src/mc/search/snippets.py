"""Snippet selection & highlighting (Story 4.4).

Picks the sentence with maximum term overlap with the query, falls back
to the chunk start on pure-semantic matches, and highlights matched
spans. All user-facing text is HTML-escaped BEFORE any <mark> is added,
so query/content can never inject markup (XSS).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from mc.search.query import normalize

_SENTENCE_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)

# CPU guard: the per-result highlight regex is an alternation of every
# distinct query word (× up to 50 results/request) — cap the word list so a
# pathological query can't amplify regex work.
MAX_HIGHLIGHT_WORDS = 32

# Tokens shorter than 3 chars carry no highlight signal in Portuguese: the
# diacritic fold turns the copula "é" into the conjunction "e", and an
# unanchored 1-char pattern would mark the letter "e" inside every word.
MIN_HIGHLIGHT_LEN = 3


@dataclass(frozen=True)
class Snippet:
    highlighted: str
    has_literal: bool


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def _content_words(text: str) -> list[str]:
    """Query words worth matching: folded words of length >= MIN_HIGHLIGHT_LEN."""
    return [w for w in _words(text) if len(w) >= MIN_HIGHLIGHT_LEN]


def build_snippet(text: str, query: str, *, max_len: int = 280) -> Snippet:
    qwords = set(_content_words(query)[:MAX_HIGHLIGHT_WORDS])
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return Snippet(highlighted=html.escape(text[:max_len]), has_literal=False)

    def overlap(sentence: str) -> int:
        return len(qwords & set(_words(sentence)))

    best_idx = 0
    best_score = 0
    for i, sentence in enumerate(sentences):
        score = overlap(sentence)
        if score > best_score:
            best_idx, best_score = i, score

    chosen = sentences[best_idx]
    has_literal = best_score > 0 and _has_literal_phrase(chosen, query)
    return Snippet(highlighted=_highlight(chosen, qwords), has_literal=has_literal)


def _has_literal_phrase(text: str, query: str) -> bool:
    if not query.strip():
        return False
    return normalize(query) in normalize(text)


def _highlight(sentence: str, qwords: set[str]) -> str:
    """Escape first, then add <mark> around whole-word query matches.

    Word boundaries are mandatory: without them a folded 1-char token ("é"
    → "e") matches the letter "e" inside every word of the snippet.
    """
    if not qwords:
        return html.escape(sentence)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(w) for w in sorted(qwords)) + r")(?!\w)",
        re.IGNORECASE,
    )

    def mark(match: re.Match) -> str:
        return f"<mark>{html.escape(match.group(0))}</mark>"

    # Escape the sentence, then highlight against the escaped text — the
    # regex operates on word characters only, so escaping first is safe.
    escaped = html.escape(sentence)
    return pattern.sub(mark, escaped)
