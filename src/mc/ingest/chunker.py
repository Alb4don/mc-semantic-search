"""Chunker (Story 3.3): 30 segments per chunk, timestamp = first segment.

Rules ported 1:1 from the v3 import script so new content is consistent
with the archive:
- flush every `size` segments (i % size == 0 with a non-empty buffer)
- chunk timestamp = first segment's start (int truncation, v3 semantics)
- url = https://www.youtube.com/watch?v=<id>&t=<int(first start)>
- texts joined with a single space
"""

from __future__ import annotations

from mc.ingest.models import Chunk

DEFAULT_SIZE = 30


def chunk_segments(
    segments: list[dict],
    video_id: str,
    *,
    size: int = DEFAULT_SIZE,
) -> list[Chunk]:
    """Group `{text,start,duration}` segments into v3-compatible chunks."""
    if size <= 0:
        raise ValueError("chunk size must be positive")

    clean = [s for s in segments if s.get("text") and s["text"].strip()]
    chunks: list[Chunk] = []
    buffer: list[dict] = []
    first_timestamp: float | None = None

    for i, line in enumerate(clean):
        if i % size == 0 and buffer:
            chunks.append(_make_chunk(video_id, buffer, first_timestamp))
            buffer = []
        if not buffer:
            first_timestamp = line["start"]
        buffer.append(line)
    if buffer:
        chunks.append(_make_chunk(video_id, buffer, first_timestamp))
    return chunks


def _make_chunk(video_id: str, buffer: list[dict], first_timestamp) -> Chunk:
    text = " ".join(str(seg["text"]) for seg in buffer)
    start = int(first_timestamp or 0)
    url = f"https://www.youtube.com/watch?v={video_id}&t={start}"
    return Chunk(video_id=video_id, text=text, url=url, start_seconds=start)
