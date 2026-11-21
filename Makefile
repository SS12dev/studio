.PHONY: setup check test api

setup:
	uv sync --python 3.12 --extra faces --extra raw --extra mac --extra extract --extra dev

check:
	uv run studio-check

test:
	uv run pytest -q

api:
	uv run uvicorn studio.api.main:app --reload --port 8000
