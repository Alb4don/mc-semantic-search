"""Result cache (Story 4.7): LRU keyed by query, freshness-invalidated.

A crawler hammering /search can't starve the embedder (NFR-1): identical
queries are served from a bounded LRU. The cache is invalidated when the
state store's last_success timestamp advances (new ingest → new archive).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("mc.search.cache")


class ResultCache:
    def __init__(
        self,
        embedder: Callable[[str], list[float]],
        state_db_path: Path,
        *,
        size: int = 128,
    ):
        self._embedder = embedder
        self._state_db_path = state_db_path
        self.size = size
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._fill_freshness: str | None = None
        self.metrics = {"hits": 0, "misses": 0}

    def get(self, query: str) -> list[float]:
        self._invalidate_if_stale()
        if query in self._data:
            self.metrics["hits"] += 1
            self._data.move_to_end(query)
            logger.debug("cache hit for %r (%s)", query, self.metrics)
            return self._data[query]
        self.metrics["misses"] += 1
        vec = self._embedder(query)
        self._data[query] = vec
        self._data.move_to_end(query)
        while len(self._data) > self.size:
            self._data.popitem(last=False)
        logger.debug("cache miss for %r (%s)", query, self.metrics)
        return vec

    def _invalidate_if_stale(self) -> None:
        current = self._read_freshness()
        if current != self._fill_freshness:
            if self._fill_freshness is not None and self._data:
                logger.info(
                    "archive freshness advanced (%s → %s) — cache cleared",
                    self._fill_freshness,
                    current,
                )
            self._fill_freshness = current
            self._data.clear()

    def _read_freshness(self) -> str | None:
        """last_success from the state store, without keeping a connection."""
        from mc.state import StateStore

        if not self._state_db_path.exists():
            return None
        try:
            store = StateStore(self._state_db_path)
            try:
                return store.last_success()
            finally:
                store.close()
        except Exception:  # unreadable state must never break search
            logger.exception("could not read freshness from state store")
            return None

    def __len__(self) -> int:
        return len(self._data)
