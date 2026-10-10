# mc-semantic-search

Semantic search over the *Morning Crypto* transcript archive.

Replaces the SearXNG + SQLite FTS5 keyword pipeline with a Qdrant-backed
hybrid (dense + sparse) search, shipped as a standalone repo and deployed
on sats4.life. Portuguese-first: users reach the **exact moment** an idea
was said, even when their wording paraphrases the transcript.

## One-command dev startup

```bash
make up        # docker compose up (qdrant + api), local ≈ production
```

Full story of the epic: `docs/tasks/mc-semantic-search-epic.md`.

## Stack

| Layer | Choice |
|---|---|
| Vector DB / search | Qdrant (native hybrid RRF) |
| Dense embeddings | `intfloat/multilingual-e5-small` (384-dim, CPU) |
| Sparse vectors | `fastembed` BM25-style |
| API | FastAPI + uvicorn |
| Frontend | Jinja2 + htmx + Alpine + Pico CSS (no Node/npm) |
| Ingestion bookkeeping | SQLite `state/ingest.db` (never on the search path) |

## Development

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
make test   # pytest + BDD
make lint   # ruff
```

Git hooks (credential guard, NFR-5) are enabled via `core.hooksPath=.githooks`.

## CLI

```bash
mc backfill --from-db extras/mc_transcripts.db   # bootstrap index from legacy DB
mc update        # daily: discover → download → chunk → embed → upsert
mc push          # Mode B: ship increment JSONL to the server
mc apply         # Mode B server side: embed + upsert queued increments
mc reindex --model <id>   # rebuild into a new collection, swap alias atomically
mc serve         # FastAPI on :8080
```

## Operations

| Doc | What |
|---|---|
| `crontab.example` | Mode A crontab (daily update, nightly snapshot, weekly apply sanity, notify hook) |
| `docs/tasks/mode-b-runbook.md` | Mode B fallback: local scrape + push, server apply, failover table |
| `docs/tasks/deploy-runbook.md` | sats4.life deployment, staging route, eval + cutover + rollback |
| `docs/tasks/cookies.md` | Cookie export/rotation + stale-cookie detection |
| `docs/tasks/backup-restore.md` | Qdrant snapshots: create/restore drill |
| `docs/tasks/eval-report.md` | Sparse-only vs hybrid MRR on known quotes (cutover evidence) |

```bash
mc eval --suite tests/eval/quotes.yaml --out docs/tasks/eval-report.md
```
