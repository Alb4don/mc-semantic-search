"""Story 2.1 — legacy SQLite FTS5 reader (mc_transcripts.db)."""

from __future__ import annotations

import csv
from pathlib import Path

from mc.ingest.legacy import count_legacy, iter_legacy_chunks
from mc.ingest.models import Chunk

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_mini.db"

URL_1234 = "https://www.youtube.com/watch?v=aqaFdI8T1lQ&t=1234"
URL_1234S = "https://www.youtube.com/watch?v=aqaFdI8T1lQ&t=1234s"
URL_20M30S = "https://www.youtube.com/watch?v=npSj_rzWrBI&t=20m30s"
URL_T0 = "https://www.youtube.com/watch?v=qT5NG1MP_d8&t=0"
URL_BROKEN = "https://example.com/not-a-youtube-url"


def test_reads_all_five_rows_four_yielded():
    chunks = list(iter_legacy_chunks(FIXTURE))
    assert len(chunks) == 4
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert {chunk.url for chunk in chunks} == {URL_1234, URL_1234S, URL_20M30S, URL_T0}


def test_start_seconds_all_three_t_forms():
    by_url = {chunk.url: chunk.start_seconds for chunk in iter_legacy_chunks(FIXTURE)}
    assert by_url[URL_1234] == 1234  # plain seconds
    assert by_url[URL_1234S] == 1234  # '1234s' form
    assert by_url[URL_20M30S] == 1230  # '20m30s' minutes+seconds form
    assert by_url[URL_T0] == 0  # t=0 stays 0


def test_chunk_fields_correct():
    chunks = {chunk.url: chunk for chunk in iter_legacy_chunks(FIXTURE)}
    chunk = chunks[URL_20M30S]
    assert chunk.video_id == "npSj_rzWrBI"
    assert chunk.text == "A conversa entre Trump e Musk sobre regulacao."
    assert chunk.url == URL_20M30S
    assert chunk.start_seconds == 1230
    assert chunk.date is None and chunk.episode is None and chunk.channel is None
    assert chunks[URL_1234].video_id == chunks[URL_1234S].video_id == "aqaFdI8T1lQ"


def test_quarantined_row_not_yielded_but_written(tmp_path: Path):
    quarantine_path = tmp_path / "quarantine.csv"
    chunks = list(iter_legacy_chunks(FIXTURE, quarantine_path=quarantine_path))
    assert all(URL_BROKEN != chunk.url for chunk in chunks)

    rows = list(csv.reader(quarantine_path.read_text().splitlines()))
    assert rows[0] == ["video_id_or_url", "reason"]
    assert len(rows) == 2
    assert rows[1] == [URL_BROKEN, "no_video_id"]


def test_no_quarantine_path_still_works():
    chunks = list(iter_legacy_chunks(FIXTURE))
    assert len(chunks) == 4


def test_count_legacy_mini_fixture():
    counts = count_legacy(FIXTURE)
    assert counts["total_rows"] == 5
    assert counts["distinct_videos"] == 3
    assert counts["parsed_chunks"] == 4
    assert counts["quarantined"] == 1
