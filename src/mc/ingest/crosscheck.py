"""Story 2.3 — cross-check chunks against the legacy transcript cache.

The legacy cache holds one transcript file per video (``<video_id>.srt``).
Classic SRT files use numbered ``-->`` timed blocks; some legacy exports
store the same captions as a JSON array of ``{"text", ...}`` objects, so
both shapes are accepted. Chunks whose words barely appear in the cached
transcript are quarantined instead of indexed.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from mc.ingest.models import Chunk

logger = logging.getLogger(__name__)

_SRT_TIME_RE = re.compile(r"\d{2}:\d{2}:\d{2}[,.]\d{3} --> \d{2}:\d{2}:\d{2}[,.]\d{3}")


@dataclass
class CrosscheckResult:
    """Outcome of verifying chunks against the legacy cache."""

    ok_chunks: list[Chunk] = field(default_factory=list)
    quarantined: list[tuple[Chunk, str]] = field(default_factory=list)
    missing_cache: set[str] = field(default_factory=set)


def _strip_accents(text: str) -> str:
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def normalize(text: str) -> str:
    """Casefold, strip accents (NFKD) and collapse whitespace."""
    return " ".join(_strip_accents(text).casefold().split())


def word_set(text: str) -> set[str]:
    """Deduplicated normalized word set."""
    return set(normalize(text).split())


def _parse_srt(raw: str) -> str:
    """Extract caption text from classic SRT blocks."""
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or _SRT_TIME_RE.search(stripped):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _parse_json_captions(raw: str) -> str | None:
    """Extract caption text from legacy JSON-array exports, if any."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    texts = [
        entry.get("text", "")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("text"), str)
    ]
    if not texts:
        return None
    return " ".join(texts)


def _transcript_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    json_text = _parse_json_captions(raw)
    return json_text if json_text is not None else _parse_srt(raw)


def _overlap(chunk_words: set[str], transcript_words: set[str]) -> float:
    if not chunk_words:
        return 0.0
    found = len(chunk_words & transcript_words)
    return found / len(chunk_words)


def verify(
    chunks: Iterable[Chunk],
    cache_dir: Path,
    *,
    threshold: float = 0.6,
) -> CrosscheckResult:
    """Verify each video's chunks against its cached transcript.

    Chunks are grouped by ``video_id``; the concatenated chunk text is
    compared with the cached transcript using normalized word-set overlap.
    Videos below ``threshold`` quarantine all their chunks; videos without
    a cache file only log a warning and pass through.
    """
    result = CrosscheckResult()

    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.video_id, []).append(chunk)

    for video_id, video_chunks in grouped.items():
        path = cache_dir / f"{video_id}.srt"
        if not path.is_file():
            logger.warning("No legacy transcript cache for video %s; skipping check", video_id)
            result.missing_cache.add(video_id)
            result.ok_chunks.extend(video_chunks)
            continue

        transcript_words = word_set(_transcript_text(path))
        chunk_words: set[str] = set()
        for chunk in video_chunks:
            chunk_words |= word_set(chunk.text)
        overlap = _overlap(chunk_words, transcript_words)

        if overlap >= threshold:
            result.ok_chunks.extend(video_chunks)
        else:
            reason = f"low_overlap:{overlap:.4f}"
            result.quarantined.extend((chunk, reason) for chunk in video_chunks)

    return result
