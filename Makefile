.PHONY: up down build test lint dev

up:       ## start qdrant + api (local ≈ production)
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build

test:     ## pytest + BDD (unit; set MC_COMPOSE_TESTS=1 for compose smoke)
	.venv/bin/python -m pytest tests/

lint:
	.venv/bin/python -m ruff check .

dev:      ## run the API from the local venv
	.venv/bin/uvicorn mc.api.app:app --host 0.0.0.0 --port 8080
