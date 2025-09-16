.DEFAULT_GOAL := all
.PHONY: build-dev python-lint lint test clean all

build-dev:
	uv sync --dev --group test
	uv pip install -e . --verbose --force-reinstall

python-lint:
	uvx ruff check .
	uvx ruff format --check .

lint: python-lint rust-lint

test:
	uv run pytest

all: build-dev format lint test

clean:
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]' `
	rm -f `find . -type f -name '*~' `
	rm -f `find . -type f -name '.*~' `
	rm -rf .cache
	rm -rf .ruff_cache
	rm -rf .venv
	rm -rf htmlcov
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf perf.data*
