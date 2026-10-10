"""Cron overlap guard (Story 3.5): flock-based, one writer per index."""

from __future__ import annotations

import fcntl
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STALE_LOCK_SECONDS = 24 * 3600


class AlreadyRunning(RuntimeError):
    """Raised when another process holds the run lock."""


class RunLock:
    """Exclusive per-run lock file (data/update.lock)."""

    def __init__(self, path: Path, *, stale_after: int = STALE_LOCK_SECONDS):
        self.path = Path(path)
        self.stale_after = stale_after
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._break_stale()
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            self._fd = None
            raise AlreadyRunning(
                f"another run holds {self.path} — refusing to overlap (cron guard)"
            ) from None

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None
            self.path.unlink(missing_ok=True)

    def _break_stale(self) -> None:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if age > self.stale_after:
            logger.warning(
                "breaking stale lock %s (age %.0fs > %ss)",
                self.path,
                age,
                self.stale_after,
            )
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()
