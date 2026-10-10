"""Shared ingestion dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """One transcript chunk (30 segments), ready to embed + index."""

    video_id: str
    text: str
    url: str
    start_seconds: int
    date: str | None = None
    episode: str | None = None
    channel: str | None = None

    def to_payload(self) -> dict:
        return {
            "text": self.text,
            "url": self.url,
            "start_seconds": self.start_seconds,
            "date": self.date,
            "episode": self.episode,
            "channel": self.channel,
            "video_id": self.video_id,
        }
