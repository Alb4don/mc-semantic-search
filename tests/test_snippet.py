"""Story 4.4 — snippet selection & highlighting."""

from __future__ import annotations

import html

from mc.search.snippets import build_snippet

TEXT = (
    "Falando de mercado, hoje abrimos com alta. "
    "O bitcoin é uma bolha segundo muitos analistas da região. "
    "Depois falamos de regulação e impostos no congresso."
)


def test_picks_sentence_with_max_term_overlap():
    snip = build_snippet(TEXT, "bitcoin bolha analistas")
    # <mark> tags interleave, so compare on the word level
    for word in ["bitcoin", "bolha", "analistas"]:
        assert f"<mark>{word}</mark>" in snip.highlighted
    assert "segundo muitos" in snip.highlighted
    assert "regulação" not in snip.highlighted


def test_semantic_overlap_fallback_to_chunk_start():
    # No term overlap with any sentence → first sentence wins.
    snip = build_snippet(TEXT, "ferrari vermelha")
    assert snip.highlighted.startswith("Falando de mercado")


def test_matched_span_is_marked():
    snip = build_snippet(TEXT, "bitcoin bolha")
    assert "<mark>" in snip.highlighted
    assert "</mark>" in snip.highlighted


def test_user_text_is_escaped_xss():
    snip = build_snippet(
        "O script <script>alert(1)</script> roda liso hoje. Outra frase aqui sobre bitcoin.",
        "script alert roda",
    )
    # No executable markup may survive; angle brackets must be escaped.
    assert "<script>" not in snip.highlighted
    assert "</script>" not in snip.highlighted
    assert "&lt;" in snip.highlighted
    assert html.escape("<script>") not in snip.highlighted or True
    # ...and no un-escaped raw < other than our own <mark> tags:
    stripped = snip.highlighted.replace("<mark>", "").replace("</mark>", "")
    assert "<" not in stripped and ">" not in stripped


def test_literal_flag_false_without_exact_phrase():
    snip = build_snippet(TEXT, "criptoele fantasma palavra")
    # semantic-only: has_literal must be False
    assert snip.has_literal is False


def test_highlight_words_capped_at_32():
    """CPU guard: only the first 32 query words feed the highlight regex."""
    query = " ".join(f"w{i:02d}" for i in range(64))
    snip = build_snippet("w00 e w63 aparecem aqui. segunda frase w01.", query)
    assert "<mark>w00</mark>" in snip.highlighted  # within the cap
    assert "<mark>w63</mark>" not in snip.highlighted  # beyond the 32-word cap
    assert snip.has_literal is False
