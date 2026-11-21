.PHONY: setup check test api

setup:
	uv sync --python 3.12

check:
	uv run studio-check

test:
	uv run pytest -q

api:
	uv run uvicorn studio.api.main:app --reload --port 8000
