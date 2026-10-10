# NFR-6: multi-stage build; model weights baked in the final image so
# container start performs no HuggingFace network access.
FROM python:3.12-slim AS builder

RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only torch first (small wheel), then the rest of the pinned deps.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    qdrant-client==1.12.0 \
    fastembed==0.7.1 \
    fastapi==0.115.12 \
    uvicorn==0.34.0 \
    jinja2==3.1.6 \
    yt-dlp==2025.6.30 \
    sentence-transformers==3.4.1 \
    pyyaml==6.0.2

FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    SENTENCE_TRANSFORMERS_HOME=/models/st \
    FASTEMBED_CACHE_DIR=/models/fastembed \
    HF_HOME=/models/hf

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/mc/ /app/src/mc/

RUN python /app/src/mc/bake_models.py

# Non-root runtime (uid 1000): host bind mounts (./data, ./state, ./snapshots)
# must be owned by this uid on the server. Models are chowned so the app can
# read them; /opt/venv stays root-owned (read-only at runtime is a plus).
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app \
    && chown -R app:app /app /models
USER app

EXPOSE 8080
CMD ["uvicorn", "mc.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
