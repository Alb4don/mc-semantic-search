"""Story 0.1 — repository layout tests.

Runs first: proves the package is importable, deps declared, lint clean,
and that credential/data paths are blocked from version control.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _mc_importable():
    import mc  # noqa: F401  (import failure = test failure)

    return mc


def test_package_importable():
    mc = _mc_importable()
    assert (REPO_ROOT / "src" / "mc").is_dir()
    assert mc.__file__ is not None


def test_pyproject_declares_deps():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    for dep in [
        "qdrant-client",
        "fastembed",
        "fastapi",
        "uvicorn",
        "jinja2",
        "yt-dlp",
        "sentence-transformers",
        "pytest",
        "pytest-bdd",
        "ruff",
    ]:
        assert dep in pyproject, f"{dep} missing from pyproject.toml"


def test_ruff_check_passes():
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_gitignore_blocks_sensitive_paths():
    gitignore = (REPO_ROOT / ".gitignore").read_text()
    for pattern in ["data/", "state/", "cookies", "*.db", "snapshots/"]:
        assert pattern in gitignore, f".gitignore must block {pattern}"


def test_pre_commit_hook_rejects_cookie_files(tmp_path):
    hook = REPO_ROOT / ".githooks" / "pre-commit"
    assert hook.exists(), ".githooks/pre-commit must exist (core.hooksPath)"
    assert hook.stat().st_mode & 0o111, "pre-commit hook must be executable"
    hook_text = hook.read_text().replace("\\", "")
    assert "www.youtube.com_cookies" in hook_text, (
        "hook must guard the cookie filename pattern"
    )

    # Behavioural check: in a throwaway clone, staging a cookie file must be
    # rejected by the hook (real git diff --cached, no fake state).
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.hooksPath", str(hook.parent)],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "config",
            "commit.gpgsign",
            "false",
        ],
        check=True,
    )
    (tmp_path / "readme.txt").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "readme.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-qm",
            "base",
        ],
        check=True,
    )
    probe = tmp_path / "www.youtube.com_cookies.txt"
    probe.write_text("# Netscape HTTP Cookie File\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "www.youtube.com_cookies.txt"], check=True)
    result = subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "-m",
            "leak cookies",
        ],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
        },
    )
    assert result.returncode != 0, "hook must reject cookie files"
    assert "cookie" in (result.stdout + result.stderr).lower()


def test_hooks_path_configured():
    result = subprocess.run(
        ["git", "config", "core.hooksPath"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert ".githooks" in result.stdout
