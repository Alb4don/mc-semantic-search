"""Story 1.5 — ingestion state store (state/ingest.db)."""

from __future__ import annotations

from pathlib import Path

from mc.state import StateStore


def _store(tmp_path: Path) -> StateStore:
    return StateStore(tmp_path / "ingest.db")


def test_mark_processed_idempotent(tmp_path):
    s = _store(tmp_path)
    s.mark_processed("vid1", run_type="backfill")
    s.mark_processed("vid1", run_type="backfill")
    assert s.is_processed("vid1")
    assert s.count_processed() == 1


def test_is_processed_false_for_unknown(tmp_path):
    s = _store(tmp_path)
    assert not s.is_processed("nope")


def test_last_success_only_via_record_success(tmp_path):
    s = _store(tmp_path)
    assert s.last_success() is None
    s.mark_processed("vid1", "update")  # must NOT set last_success
    assert s.last_success() is None
    s.record_success("update")
    ts = s.last_success()
    assert ts is not None
    s.record_success("update")
    assert s.last_success() >= ts


def test_wal_mode_and_busy_timeout(tmp_path):
    s = _store(tmp_path)
    s.mark_processed("vid1", "update")
    assert s.journal_mode() == "wal"
    assert s.busy_timeout_ms() > 0


def test_corrupt_db_recreated_with_warning(tmp_path):
    db = tmp_path / "ingest.db"
    db.write_bytes(b"not a sqlite database at all" * 10)
    s = StateStore(db)
    s.mark_processed("vid1", "backfill")  # must work despite corruption
    assert s.is_processed("vid1")


def test_missing_db_created(tmp_path):
    s = _store(tmp_path / "subdir" / "ingest.db")
    assert s.is_processed("x") is False


def test_processed_persisted(tmp_path):
    s1 = _store(tmp_path)
    s1.mark_processed("vid1", "update")
    s1.close()
    s2 = StateStore(tmp_path / "ingest.db")
    assert s2.is_processed("vid1")
