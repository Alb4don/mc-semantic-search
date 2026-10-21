"""Story 0.3 — CLI entrypoints."""

from __future__ import annotations

import subprocess
import sys

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


def _run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "mc.cli", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_help_lists_all_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    for cmd in [
        "backfill",
        "update",
        "push",
        "apply",
        "reindex",
        "serve",
        "stats-backfill",
    ]:
        assert cmd in result.stdout, f"missing subcommand {cmd}"


def test_unknown_subcommand_exits_nonzero_with_usage():
    result = _run_cli("definitely-not-a-command")
    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()


def test_stats_backfill_rejects_unknown_flag():
    """Story 6.4: a bad flag on a real subcommand still exits non-zero."""
    result = _run_cli("stats-backfill", "--not-a-real-flag")
    assert result.returncode != 0
    assert "usage" in (result.stdout + result.stderr).lower()
