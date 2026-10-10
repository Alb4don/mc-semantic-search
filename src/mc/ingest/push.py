"""Daily increment push (Story 3.6): ship data/queue/pending.jsonl to Mode B.

Reads the pending increment written by `mc update`, filters out records
already pushed (bookkeeping is per CONTENT HASH of the chunk record, not per
video), writes the unpushed set to a dated JSONL file and ships it to the
destination with rsync. Push markers are recorded only after a successful
ship, so a failed run leaves the dated file behind for retry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

from mc import config
from mc.ingest.lock import RunLock
from mc.state import StateStore

logger = logging.getLogger(__name__)

_PENDING_NAME = "pending.jsonl"
_LOCK_NAME = "update.lock"  # shared with `mc update` so they never overlap
_RSYNC_FLAGS = ("-a", "--checksum")


def _content_hash(record: dict) -> str:
    """sha256 of the canonical JSON dump — the push bookkeeping key."""
    payload = json.dumps(record, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_shipper(dest: str) -> Callable[[Path], None]:
    """Return a callable shipping a file to `dest` via rsync (real transport)."""

    def ship(path: Path) -> None:
        subprocess.run(["rsync", *_RSYNC_FLAGS, str(path), dest], check=True)

    return ship


def _default_shipper(dest: str | None) -> Callable[[Path], None] | None:
    dest = dest or os.environ.get("MC_PUSH_DEST")
    if not dest:
        return None
    return build_shipper(dest)


def _read_pending(pending_path: Path) -> list[dict]:
    """Parse pending.jsonl; corrupt lines are skipped with a warning."""
    if not pending_path.exists():
        return []
    records: list[dict] = []
    for lineno, raw in enumerate(
        pending_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s line %d: invalid JSON — skipped", pending_path, lineno)
            continue
        if not isinstance(record, dict):
            logger.warning(
                "%s line %d: not a JSON object — skipped", pending_path, lineno
            )
            continue
        records.append(record)
    return records


def _ship_date(now: date | datetime | None) -> str:
    if now is None:
        return date.today().isoformat()
    return now.isoformat()[:10]


def run_push(
    *,
    queue_dir: Path | None = None,
    state_path: Path | None = None,
    shipper: Callable[[Path], None] | None = None,
    dest: str | None = None,
    now: date | None = None,
) -> int:
    """Ship the daily increment. Returns a process exit code."""
    queue_dir = Path(queue_dir) if queue_dir is not None else config.QUEUE_DIR
    state_path = Path(state_path) if state_path is not None else config.INGEST_DB_PATH
    pending_path = queue_dir / _PENDING_NAME
    lock = RunLock(queue_dir.parent / _LOCK_NAME)

    with lock:  # AlreadyRunning propagates to the caller (cron guard)
        state = StateStore(state_path)
        records = _read_pending(pending_path)
        unpushed: list[tuple[dict, str]] = []
        for record in records:
            content_hash = _content_hash(record)
            if state.get_key(f"pushed:{content_hash}") is None:
                unpushed.append((record, content_hash))

        if not unpushed:
            print("nothing to push")
            return 0

        out_path = queue_dir / f"{_ship_date(now)}.jsonl"
        out_path.write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r, _ in unpushed),
            encoding="utf-8",
        )

        if shipper is None:
            shipper = _default_shipper(dest)
            if shipper is None:
                print(
                    "MC_PUSH_DEST is not set (e.g. 'server:~/mc/queue/') — "
                    f"cannot ship; {out_path} left for retry",
                    file=sys.stderr,
                )
                return 2

        try:
            shipper(out_path)
        except (subprocess.CalledProcessError, OSError) as exc:
            logger.error("ship failed (%s) — %s left for retry", exc, out_path)
            return 1

        for _record, content_hash in unpushed:
            state.record_key(f"pushed:{content_hash}", _now_iso())
        return 0


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
