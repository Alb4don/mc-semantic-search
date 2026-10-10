"""Story 3.5 — cron overlap guard (flock-based)."""

from __future__ import annotations

import os
import time

import pytest

from mc.ingest.lock import AlreadyRunning, RunLock

LOCK_PATH = "update.lock"


def test_second_acquire_raises(tmp_path):
    lock_path = tmp_path / LOCK_PATH
    with RunLock(lock_path):
        with pytest.raises(AlreadyRunning):
            RunLock(lock_path).acquire()


def test_release_allows_reacquire(tmp_path):
    lock_path = tmp_path / LOCK_PATH
    lock = RunLock(lock_path)
    lock.acquire()
    lock.release()
    with RunLock(lock_path):  # no error
        pass


def test_context_manager(tmp_path):
    lock_path = tmp_path / LOCK_PATH
    with RunLock(lock_path):
        with pytest.raises(AlreadyRunning):
            RunLock(lock_path).acquire()
    with RunLock(lock_path):
        pass


def test_stale_lock_older_than_24h_is_broken_with_warning(tmp_path, caplog):
    import logging

    lock_path = tmp_path / LOCK_PATH
    lock_path.touch()
    stale_age = 25 * 3600
    os.utime(lock_path, (time.time() - stale_age, time.time() - stale_age))

    with caplog.at_level(logging.WARNING):
        with RunLock(lock_path):  # must not raise
            pass
    assert any("stale" in r.message.lower() for r in caplog.records)


def test_fresh_stale_check(tmp_path):
    lock_path = tmp_path / LOCK_PATH
    with RunLock(lock_path):
        # A lock held right now (mtime fresh) must not be considered stale.
        with pytest.raises(AlreadyRunning):
            RunLock(lock_path).acquire()


def test_symlink_at_lock_path_is_not_followed(tmp_path):
    """O_NOFOLLOW: a symlink planted at the lock path must not be opened."""
    target = tmp_path / "victim.txt"
    target.write_text("")
    symlink = tmp_path / LOCK_PATH
    symlink.symlink_to(target)

    with pytest.raises(OSError):  # ELOOP from O_NOFOLLOW
        RunLock(symlink).acquire()

    assert target.read_text() == ""  # symlink target untouched
