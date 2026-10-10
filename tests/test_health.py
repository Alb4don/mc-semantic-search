"""Story 4.6 — /health with archive freshness.

Unit-level: qdrant reachability is exercised through fakes (no container
needed), freshness through a tmp StateStore. The threshold comes from the
injected freshness_max_hours (env FRESHNESS_MAX_HOURS via config).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mc.api.app import create_app
from mc.api.health import health_snapshot
from mc.state import StateStore


class ReachableClient:
    def get_collections(self):
        return []


class UnreachableClient:
    def get_collections(self):
        raise ConnectionError("connection refused")


def _iso(hours_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours_ago)).isoformat(
        timespec="seconds"
    )


def _fresh(ts: str) -> bool:
    """True when an ISO timestamp is within 60s of now (second-tick safe)."""
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return abs(datetime.now(UTC) - parsed) < timedelta(seconds=60)


@pytest.fixture
def state(tmp_path):
    return StateStore(tmp_path / "ingest.db")


# --- health_snapshot ---------------------------------------------------------


def test_ok_when_qdrant_reachable_and_update_is_fresh(state, monkeypatch):
    state.record_success("update")
    monkeypatch.setattr(
        "mc.api.health.read_meta_fingerprint", lambda c, n: {"dense": "abc"}
    )
    code, body = health_snapshot(
        ReachableClient(), state, 72, collection="coll"
    )
    assert code == 200
    assert body["status"] == "ok"
    assert _fresh(body["last_update"])
    assert body["fingerprint"] == {"dense": "abc"}
    assert body["qdrant"] == "reachable"


def test_last_update_null_when_never_ingested(state):
    code, body = health_snapshot(ReachableClient(), state, 72)
    assert code == 200
    assert body["status"] == "ok"
    assert body["last_update"] is None
    assert body["fingerprint"] is None


def test_stale_when_last_success_older_than_threshold(state, monkeypatch):
    state.record_key("last_success:update", _iso(hours_ago=100))
    monkeypatch.setattr("mc.api.health.read_meta_fingerprint", lambda c, n: None)
    code, body = health_snapshot(ReachableClient(), state, 72)
    assert code == 200
    assert body["status"] == "stale"


def test_threshold_is_injected_not_hardcoded(state):
    state.record_key("last_success:update", _iso(hours_ago=100))
    assert health_snapshot(ReachableClient(), state, 200)[1]["status"] == "ok"
    assert health_snapshot(ReachableClient(), state, 24)[1]["status"] == "stale"


def test_503_when_qdrant_unreachable(state):
    code, body = health_snapshot(UnreachableClient(), state, 72)
    assert code == 503
    assert body["status"] == "unreachable"
    # No exception text in the body — internal qdrant URLs must not leak.
    assert "detail" not in body
    assert "connection refused" not in str(body)


# --- /health endpoint (thin wrapper over the snapshot) -----------------------


def test_endpoint_ok(state, monkeypatch):
    state.record_success("update")
    monkeypatch.setattr(
        "mc.api.health.read_meta_fingerprint", lambda c, n: {"dense": "abc"}
    )
    app = create_app(client=ReachableClient(), state=state, freshness_max_hours=72)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert _fresh(body["last_update"])
    assert body["fingerprint"] == {"dense": "abc"}
    assert body["qdrant"] == "reachable"


def test_endpoint_stale(state):
    state.record_key("last_success:update", _iso(hours_ago=100))
    app = create_app(client=ReachableClient(), state=state, freshness_max_hours=72)
    body = TestClient(app).get("/health").json()
    assert body["status"] == "stale"


def test_endpoint_503_with_unreachable_body(state):
    app = create_app(client=UnreachableClient(), state=state, freshness_max_hours=72)
    response = TestClient(app).get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unreachable"
    assert "detail" not in response.json()
