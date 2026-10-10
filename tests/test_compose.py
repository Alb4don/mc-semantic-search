"""Story 0.2 — Docker Compose smoke tests.

Heavy: requires a built image (docker compose build) and a running stack.
Skipped unless MC_COMPOSE_TESTS=1 so `make test` stays fast on dev boxes;
CI and pre-release runs set the env var.
"""

import os
import subprocess
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    os.environ.get("MC_COMPOSE_TESTS") != "1",
    reason="set MC_COMPOSE_TESTS=1 to run docker compose smoke tests",
)


def _compose(*args):
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def _compose_config_yaml():
    # Validate structure without needing a running daemon for every check.
    result = _compose("config", "--format", "yaml")
    return result.stdout


def test_compose_config_defines_qdrant_and_api_with_mem_limit():
    cfg = _compose_config_yaml()
    assert "mem_limit" in cfg, "every service must declare mem_limit (NFR-2)"
    for service in ["qdrant", "api"]:
        assert f"{service}:" in cfg


def test_compose_up_api_answers_health():
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    try:
        deadline = 30
        ok = False
        import time

        for _ in range(deadline * 2):
            try:
                with urllib.request.urlopen(
                    "http://localhost:8080/health", timeout=2
                ) as resp:
                    ok = resp.status == 200
                    break
            except Exception:
                time.sleep(0.5)
        assert ok, "api /health must answer within 30s of compose up"
    finally:
        _compose("down")


def test_qdrant_volume_dir_created():
    _compose("up", "-d", "qdrant")
    try:
        deadline = 30
        import time

        for _ in range(deadline * 2):
            if (REPO_ROOT / "data" / "qdrant").is_dir():
                break
            time.sleep(0.5)
        assert (REPO_ROOT / "data" / "qdrant").is_dir()
    finally:
        _compose("down")


def test_dockerfile_bakes_model_weights():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "multilingual-e5-small" in dockerfile or (
        "bake_models" in dockerfile
    ), "model weights must be baked into the image (NFR-6)"
    # fastembed sparse encoder weights baked too
    assert "fastembed" in dockerfile.lower() or "bake_models" in dockerfile
