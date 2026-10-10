"""Story 2.3 — legacy cache cross-check (transcript overlap quarantine)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import pytest

from mc.ingest.crosscheck import CrosscheckResult, verify
from mc.ingest.models import Chunk

FIXTURES = Path(__file__).parent / "fixtures" / "crosscheck"


def _chunk(video_id: str, text: str) -> Chunk:
    return Chunk(
        video_id=video_id,
        text=text,
        url=f"https://www.youtube.com/watch?v={video_id}",
        start_seconds=0,
    )


@pytest.fixture
def match_chunk() -> Chunk:
    return _chunk(
        "vid_match",
        "Olá pessoal, hoje vamos falar sobre a mineração de Bitcoin "
        "e o consumo de energia das mineradoras. "
        "O relatório mostra que a rede usa um mix de energia sustentável.",
    )


@pytest.fixture
def mismatch_chunk() -> Chunk:
    return _chunk(
        "vid_mismatch",
        "O mercado de criptomoedas entrou em queda livre hoje "
        "e os investidores temem o urso.",
    )


@pytest.fixture
def missing_chunk() -> Chunk:
    return _chunk("vid_missing", "Texto qualquer sem cache em disco.")


def test_matching_video_passes(match_chunk: Chunk) -> None:
    result = verify([match_chunk], FIXTURES)

    assert isinstance(result, CrosscheckResult)
    assert result.ok_chunks == [match_chunk]
    assert result.quarantined == []
    assert result.missing_cache == set()


def test_mismatched_video_is_quarantined(mismatch_chunk: Chunk) -> None:
    result = verify([mismatch_chunk], FIXTURES)

    assert result.ok_chunks == []
    assert result.missing_cache == set()
    (chunk, reason) = result.quarantined[0]
    assert chunk == mismatch_chunk
    assert reason.startswith("low_overlap")


def test_missing_cache_file_warns_but_passes(
    missing_chunk: Chunk, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="mc.ingest.crosscheck"):
        result = verify([missing_chunk], FIXTURES)

    assert result.missing_cache == {"vid_missing"}
    assert result.ok_chunks == [missing_chunk]
    assert result.quarantined == []
    assert "vid_missing" in caplog.text


def test_threshold_override_allows_everything(
    match_chunk: Chunk, mismatch_chunk: Chunk, missing_chunk: Chunk
) -> None:
    result = verify(
        [mismatch_chunk, match_chunk, missing_chunk], FIXTURES, threshold=0
    )

    assert result.ok_chunks == [mismatch_chunk, match_chunk, missing_chunk]
    assert result.quarantined == []
    assert result.missing_cache == {"vid_missing"}


def test_verify_does_not_mutate_chunks(
    match_chunk: Chunk, mismatch_chunk: Chunk, missing_chunk: Chunk
) -> None:
    chunks = [match_chunk, mismatch_chunk, missing_chunk]
    snapshots = [replace(chunk) for chunk in chunks]

    verify(chunks, FIXTURES)

    assert chunks == snapshots
