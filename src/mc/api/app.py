"""FastAPI application factory.

Story 0.2: minimal liveness so `docker compose up` serves /health.
Theme 4 stories extend this module with /search, full /health freshness,
the Jinja UI and the embedded player.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="mc-semantic-search", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
