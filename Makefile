.PHONY: setup check test build app dev api web

setup:            ## install Python and frontend dependencies
	uv sync --python 3.12
	cd frontend && npm install

check:
	uv run studio-check

test:
	uv run pytest -q

build:            ## build the React app into frontend/dist (served by the API)
	cd frontend && npm run build

app: build        ## build, then run the app and open it in the browser
	uv run studio serve

dev:              ## hot-reload: API on :8000 + Vite on :5173 (open :5173)
	$(MAKE) -j2 api web

api:
	uv run uvicorn --factory studio.api.main:create_app --reload --port 8000 --reload-dir backend

web:
	cd frontend && npm run dev
