.PHONY: dev run worker install install-dev test test-unit test-coverage lint format docker-build docker-up download-models ensure-venv ensure-runtime ensure-dev

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

ensure-venv:
	@test -x "$(VENV_PYTHON)" || $(PYTHON) -m venv "$(VENV)"

ensure-runtime: ensure-venv
	@$(VENV_PYTHON) -c "import fastapi, uvicorn" >/dev/null 2>&1 || { \
		$(VENV_PYTHON) -m pip install --upgrade pip && \
		$(VENV_PIP) install -r requirements.txt; \
	}

ensure-dev: ensure-venv
	@$(VENV_PYTHON) -c "import fastapi, uvicorn" >/dev/null 2>&1 || { \
		$(VENV_PYTHON) -m pip install --upgrade pip && \
		$(VENV_PIP) install -r requirements-dev.txt; \
	}
	@$(VENV_PYTHON) -m pytest --version >/dev/null 2>&1
	@$(VENV_PYTHON) -m ruff --version >/dev/null 2>&1

run: ensure-runtime
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5051}

dev: ensure-dev
	$(VENV_PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port $${PORT:-5051} --reload

worker: ensure-runtime
	$(VENV_PYTHON) -m arq src.worker.WorkerSettings

install: ensure-venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements.txt

install-dev: ensure-venv
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt

test: ensure-dev
	$(VENV_PYTHON) -m pytest tests/ -v

test-unit: ensure-dev
	$(VENV_PYTHON) -m pytest tests/unit/ -v

test-coverage: ensure-dev
	$(VENV_PYTHON) -m coverage run -m pytest tests/ -v
	$(VENV_PYTHON) -m coverage report -m
	$(VENV_PYTHON) -m coverage html

lint: ensure-dev
	$(VENV_PYTHON) -m ruff check src/ tests/

format: ensure-dev
	$(VENV_PYTHON) -m ruff format src/ tests/

download-models: ensure-runtime
	$(VENV_PYTHON) scripts/download_models.py

docker-build:
	docker build -t media-service .

docker-up:
	docker compose up --build
