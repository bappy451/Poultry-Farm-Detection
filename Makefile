PYTHON_BOOTSTRAP ?= python3.12
PYTHON := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: venv install install-ml test

venv:
	$(PYTHON_BOOTSTRAP) -m venv .venv

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-ml: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-ml.txt

test:
	$(PYTHON) -m pytest
