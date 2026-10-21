"""Story 3.6 — daily increment push (Mode B ship step)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from mc.ingest.lock import AlreadyRunning, RunLock
from mc.ingest.push import build_shipper, run_push
from mc.state import StateStore


def _record(video_id: str, text: str = "chunk text") -> dict:
    return {
        "video_id": video_id,
        "text": text,
        "url": f"https://youtube.com/watch?v={video_id}",
        "start_seconds": 0,
        "date": "2026-10-09",
        "episode": "ep",
        "channel": "Morning Crypto",
    }


def _content_hash(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()


class FakeShipper:
    """Captures shipped file paths + the destination it was built with."""

    def __init__(self) -> None:
        self.dest = "fake:~/mc/queue/"
        self.paths: list[Path] = []

    def __call__(self, path: Path) -> None:
        self.paths.append(path)


@pytest.fixture
def queue_dir(tmp_path) -> Path:
    d = tmp_path / "queue"
    d.mkdir()
    return d


@pytest.fixture
def state(tmp_path) -> StateStore:
    return StateStore(tmp_path / "ingest.db")


def _write_pending(queue_dir: Path, records: list[dict], raw_lines: list[str]) -> None:
    lines = [json.dumps(r) for r in records] + raw_lines
    (queue_dir / "pending.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_push_filters_pushed_marks_hashes_and_ships_once(
    queue_dir, state, tmp_path
):
    records = [_record("vid1"), _record("vid2"), _record("vid3")]
    _write_pending(queue_dir, records, raw_lines=[])
    state.record_key(f"pushed:{_content_hash(records[1])}", "2026-10-09T00:00:00+00:00")

    shipper = FakeShipper()
    exit_code = run_push(
        queue_dir=queue_dir,
        state_path=state.path,
        shipper=shipper,
        now=date(2026, 10, 10),
    )

    assert exit_code == 0
    assert len(shipper.paths) == 1
    assert shipper.paths[0].name.startswith("2026-10-10-"), (
        "shipment file must be date-prefixed and per-run unique"
    )

    shipped = [
        json.loads(line)
        for line in shipper.paths[0].read_text().splitlines()
    ]
    assert shipped == [records[0], records[2]]

    assert state.get_key(f"pushed:{_content_hash(records[0])}") is not None
    assert state.get_key(f"pushed:{_content_hash(records[2])}") is not None
    assert state.get_key(f"pushed:{_content_hash(records[1])}") is not None


def test_second_run_is_a_noop(queue_dir, state):
    records = [_record("vid1"), _record("vid2")]
    _write_pending(queue_dir, records, raw_lines=[])

    shipper = FakeShipper()
    first = run_push(
        queue_dir=queue_dir,
        state_path=state.path,
        shipper=shipper,
        now=date(2026, 10, 10),
    )
    assert first == 0

    exit_code = run_push(
        queue_dir=queue_dir,
        state_path=state.path,
        shipper=shipper,
        now=date(2026, 10, 11),
    )

    assert exit_code == 0
    assert len(shipper.paths) == 1  # not shipped again
    assert not list(queue_dir.glob("2026-10-11-*.jsonl"))  # no new shipment
    assert [p.name for p in queue_dir.glob("2026-10-10-*.jsonl")] == [
        shipper.paths[0].name
    ]


def test_nothing_to_push_prints_message(queue_dir, state, capsys):
    exit_code = run_push(
        queue_dir=queue_dir, state_path=state.path, shipper=FakeShipper()
    )
    assert exit_code == 0
    assert "nothing to push" in capsys.readouterr().out


def test_empty_pending_jsonl_is_noop(queue_dir, state):
    (queue_dir / "pending.jsonl").write_text("", encoding="utf-8")
    shipper = FakeShipper()
    exit_code = run_push(
        queue_dir=queue_dir, state_path=state.path, shipper=shipper
    )
    assert exit_code == 0
    assert shipper.paths == []
    assert [p for p in queue_dir.glob("*.jsonl") if p.name != "pending.jsonl"] == []


def test_missing_pending_jsonl_is_noop(queue_dir, state):
    shipper = FakeShipper()
    exit_code = run_push(
        queue_dir=queue_dir, state_path=state.path, shipper=shipper
    )
    assert exit_code == 0
    assert shipper.paths == []
    assert [p for p in queue_dir.glob("*.jsonl") if p.name != "pending.jsonl"] == []


def test_corrupt_json_line_skipped_with_warning(queue_dir, state, caplog):
    import logging

    records = [_record("vid1"), _record("vid2")]
    _write_pending(queue_dir, records, raw_lines=["{not valid json"])

    shipper = FakeShipper()
    with caplog.at_level(logging.WARNING):
        exit_code = run_push(
            queue_dir=queue_dir,
            state_path=state.path,
            shipper=shipper,
            now=date(2026, 10, 10),
        )

    assert exit_code == 0
    assert any("invalid json" in r.message.lower() for r in caplog.records)
    shipped = [
        json.loads(line)
        for line in shipper.paths[0].read_text().splitlines()
    ]
    assert shipped == records


def test_missing_destination_leaves_file_for_retry(queue_dir, state, monkeypatch):
    _write_pending(queue_dir, [_record("vid1")], raw_lines=[])
    monkeypatch.delenv("MC_PUSH_DEST", raising=False)

    exit_code = run_push(queue_dir=queue_dir, state_path=state.path)

    assert exit_code != 0
    # record NOT marked pushed and per-run file left on disk for the retry
    record = json.loads((queue_dir / "pending.jsonl").read_text().splitlines()[0])
    assert state.get_key(f"pushed:{_content_hash(record)}") is None
    leftovers = [p for p in queue_dir.glob(f"{date.today().isoformat()}-*.jsonl")]
    assert len(leftovers) == 1


def test_lock_conflict_raises_already_running(queue_dir, state):
    lock_path = queue_dir.parent / "update.lock"
    held = RunLock(lock_path)
    held.acquire()
    try:
        with pytest.raises(AlreadyRunning):
            run_push(
                queue_dir=queue_dir, state_path=state.path, shipper=FakeShipper()
            )
    finally:
        held.release()


def test_build_shipper_runs_rsync(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["check"] = kwargs.get("check")

    monkeypatch.setattr(subprocess, "run", fake_run)
    shipper = build_shipper("server:~/mc/queue/")
    shipper(tmp_path / "2026-10-10.jsonl")

    assert captured["cmd"] == [
        "rsync",
        "-a",
        "--checksum",
        "--",
        str(tmp_path / "2026-10-10.jsonl"),
        "server:~/mc/queue/",
    ]
    assert captured["check"] is True


def test_second_push_same_day_never_overwrites_first(queue_dir, state):
    """Review fix: same-day reruns write a NEW per-run file — rsyncing a
    truncated {date}.jsonl over the remote file destroyed the morning batch."""
    first_records = [_record("vid1")]
    _write_pending(queue_dir, first_records, raw_lines=[])
    shipper = FakeShipper()
    assert (
        run_push(
            queue_dir=queue_dir,
            state_path=state.path,
            shipper=shipper,
            now=date(2026, 10, 10),
        )
        == 0
    )
    first_path = shipper.paths[0]
    first_content = first_path.read_text()

    # more records arrive the same day; a second push runs
    later_records = [_record("vid2")]
    _write_pending(queue_dir, later_records, raw_lines=[])
    exit_code = run_push(
        queue_dir=queue_dir,
        state_path=state.path,
        shipper=shipper,
        now=date(2026, 10, 10),
    )

    assert exit_code == 0
    assert len(shipper.paths) == 2, "second same-day push must ship a new file"
    assert shipper.paths[1] != first_path
    assert first_path.read_text() == first_content, (
        "the first shipment file must be untouched"
    )
    shipped_second = [json.loads(line) for line in shipper.paths[1].read_text().splitlines()]
    assert shipped_second == later_records
