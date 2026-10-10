"""Story 3.3 — chunker: 30 segments, timestamp = first segment (v3 parity)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mc.ingest.chunker import chunk_segments


def _seg(i: int, text: str | None = None) -> dict:
    return {
        "text": text if text is not None else f"segmento número {i}",
        "start": float(i * 5),
        "duration": 4.5,
    }


def test_75_segments_make_3_chunks():
    segments = [_seg(i) for i in range(75)]
    chunks = chunk_segments(segments, video_id="abc")
    assert len(chunks) == 3
    assert [len_ for len_ in _chunk_lengths(chunks)] == [30, 30, 15]


def _chunk_lengths(chunks):
    return [c.text.count("segmento número") for c in chunks]


def test_chunk_url_uses_first_segment_start():
    segments = [_seg(i) for i in range(35)]
    chunks = chunk_segments(segments, video_id="abc")
    # second chunk starts at segment 30 → start=150s
    assert chunks[1].start_seconds == 150
    assert chunks[1].url.endswith("&t=150")


def test_empty_and_whitespace_segments_skipped():
    segments = [
        _seg(0, "texto real um"),
        _seg(1, "   "),
        _seg(2, ""),
        _seg(3, "texto real dois"),
    ]
    chunks = chunk_segments(segments, video_id="abc")
    assert len(chunks) == 1
    assert chunks[0].text == "texto real um texto real dois"
    assert chunks[0].start_seconds == 0


def test_unicode_safe_joining():
    segments = [_seg(0, "olá ção µñana épéli"), _seg(1, "segundo texto — ok")]
    chunks = chunk_segments(segments, video_id="abc")
    assert "olá ção µñana épéli segundo texto — ok" == chunks[0].text


def test_golden_vs_v3_logic():
    """Golden test: chunks must match what import_transcriptions_v3.py produced.

    The fixture stores segments plus the chunk (text,url) pairs computed by
    the v3 algorithm (30-segment flush, first-timestamp url, space join).
    """
    golden = json.loads(
        (Path(__file__).parent / "fixtures" / "chunker_golden.json").read_text()
    )
    chunks = chunk_segments(golden["segments"], video_id=golden["video_id"])
    assert [(c.text, c.url) for c in chunks] == [
        (text, url) for text, url in golden["chunks"]
    ]


def test_default_size_is_30():
    segments = [_seg(i) for i in range(61)]
    chunks = chunk_segments(segments, video_id="abc")
    assert len(chunks) == 3  # 30/30/1
    with pytest.raises(ValueError):
        chunk_segments(segments, video_id="abc", size=0)
