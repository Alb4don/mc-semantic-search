"""Query normalization & verbatim-phrase detection (Story 4.1).

Sparse vectors get PLAIN TEXT only — no boolean operators can leak into
the encoder. Normalization (casefold, diacritic folding) exists solely
for matching; the raw query is preserved for display.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_QUOTE_RE = re.compile(r'"([^"]+)"')
_CTRL_CHARS_RE = re.compile(r"[\[\]{}()<>~^*?:\\/+!]")

# Characters stripped from the encoder-facing text: sparse analyzers can
# misinterpret operator-ish punctuation. Word-level operators are dropped
# entirely so they can't inflate term importance.
_BOOL_WORDS = {"and", "or", "not", "site", "intitle", "incollection", "inurl"}


@dataclass(frozen=True)
class Query:
    raw: str
    normalized: str
    is_verbatim: bool
    phrase: str | None
    normalized_phrase: str | None


def normalize(text: str) -> str:
    """Casefold + diacritic-fold + collapse whitespace (matching only)."""
    folded = unicodedata.normalize("NFKD", text)
    without = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(without.casefold().split())


def parse_query(raw: str) -> Query:
    raw = raw.strip()
    match = _QUOTE_RE.search(raw)
    is_verbatim = match is not None
    phrase = match.group(1).strip() if match else None

    # Encoder-facing text: unquote, drop operator chars, drop boolean words.
    flat = _QUOTE_RE.sub(r"\1", raw)
    flat = _CTRL_CHARS_RE.sub(" ", flat)
    words = [
        w for w in normalize(flat).split() if w.strip(".-") not in _BOOL_WORDS
    ]
    normalized = " ".join(words) if words else normalize(flat)

    return Query(
        raw=raw,
        normalized=normalized,
        is_verbatim=is_verbatim,
        phrase=phrase,
        normalized_phrase=normalize(phrase) if phrase else None,
    )
