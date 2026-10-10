"""Health snapshot with archive freshness (Story 4.6).

The /health route stays a thin wrapper: all reachability + freshness
logic lives in `health_snapshot` so it can be unit-tested without HTTP.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from qdrant_client import QdrantClient

from mc.qdrant import read_meta_fingerprint
from mc.state import StateStore

logger = logging.getLogger(__name__)

LAST_SUCCESS_KEY = "update"  # run_type recorded by the ingest update flow


def _hours_since(ts: str | None) -> float | None:
    """Hours since an ISO timestamp, or None when absent/unparseable."""
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return (datetime.now(UTC) - parsed).total_seconds() / 3600
    except (ValueError, TypeError, OverflowError):
        return None


def health_snapshot(
    client: QdrantClient,
    state: StateStore,
    freshness_max_hours: int,
    *,
    collection: str | None = None,
) -> tuple[int, dict]:
    """Probe qdrant + the state store; return (http status, json body)."""
    try:
        client.get_collections()
    except Exception as exc:  # any transport error → qdrant is down
        # str(exc) would disclose internal qdrant host/URLs to callers.
        logger.warning("qdrant unreachable: %s", exc)
        return 503, {"status": "unreachable"}

    last_update = state.last_success(LAST_SUCCESS_KEY)
    age_hours = _hours_since(last_update)
    status = "ok"
    if age_hours is not None and age_hours > freshness_max_hours:
        status = "stale"

    fingerprint = None
    if collection is not None:
        try:
            fingerprint = read_meta_fingerprint(client, collection)
        except Exception:  # missing/unreadable meta point is not unhealthy
            fingerprint = None

    return 200, {
        "status": status,
        "last_update": last_update,
        "fingerprint": fingerprint,
        "qdrant": "reachable",
    }
