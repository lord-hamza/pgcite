PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
QUESTION ?= How far back can we look in the webhook delivery logs compared with the API request log?

.PHONY: install db down ingest serve test lint eval ablation baseline bench demo

install:
	$(PYTHON) -m pip install -e ".[dev]"

db:
	docker compose up -d --wait db

down:
	docker compose down

ingest:
	$(PYTHON) scripts/ingest.py

serve:
	$(PYTHON) -m uvicorn pgcite.api:app --reload

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

eval:
	$(PYTHON) -m evals.run

ablation:
	$(PYTHON) -m evals.run --ablation

baseline:
	$(PYTHON) -m evals.run --ablation --update-baseline

bench:
	$(PYTHON) scripts/bench.py

demo:
	docker compose up -d --build --wait
	docker compose exec api python scripts/ingest.py
	curl -s localhost:8000/ask -H "Content-Type: application/json" \
		-d '{"question": "$(QUESTION)"}' | python3 -m json.tool
