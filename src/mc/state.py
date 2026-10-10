"""Ingestion state store: durable bookkeeping in state/ingest.db (Story 1.5).

Never on the search path — only ingest/bookkeeping code touches it. WAL
mode + busy_timeout so a cron run never blocks the API (which doesn't
read this db anyway) or another writer.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS processed (
    video_id TEXT NOT NULL,
    run_type TEXT NOT NULL,
    ts TEXT NOT NULL,
    PRIMARY KEY (video_id)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_BUSY_TIMEOUT_MS = 5000


class StateStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = None
        self._open()

    def _open(self) -> None:
        try:
            self._conn = self._connect()
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.DatabaseError:
            logger.warning(
                "state db %s is corrupt — recreating from scratch", self.path
            )
            self.path.unlink(missing_ok=True)
            self._conn = self._connect()
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        return conn

    # --- processed videos ---------------------------------------------------
    def mark_processed(self, video_id: str, run_type: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO processed(video_id, run_type, ts) VALUES (?, ?, ?) "
                "ON CONFLICT(video_id) DO UPDATE SET run_type=excluded.run_type, "
                "ts=excluded.ts",
                (video_id, run_type, _now_iso()),
            )

    def is_processed(self, video_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM processed WHERE video_id = ?", (video_id,)
        ).fetchone()
        return row is not None

    def count_processed(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM processed").fetchone()[0]

    # --- run success bookkeeping (NFR-4) -------------------------------------
    def record_success(self, run_type: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (f"last_success:{run_type}", _now_iso()),
            )

    def last_success(self, run_type: str = "update") -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (f"last_success:{run_type}",)
        ).fetchone()
        return row[0] if row else None

    # --- push bookkeeping (Story 3.6) -----------------------------------------
    def mark_pushed(self, video_id: str) -> None:
        self.record_key(f"pushed:{video_id}", _now_iso())

    def is_pushed(self, video_id: str) -> bool:
        return self.get_key(f"pushed:{video_id}") is not None

    # --- generic meta ---------------------------------------------------------
    def record_key(self, key: str, value: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_key(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    # --- diagnostics ------------------------------------------------------------
    def journal_mode(self) -> str:
        return self._conn.execute("PRAGMA journal_mode").fetchone()[0]

    def busy_timeout_ms(self) -> int:
        return self._conn.execute("PRAGMA busy_timeout").fetchone()[0]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
