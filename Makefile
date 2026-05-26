# Unified Task Runner for esther_nlp pipeline.

.PHONY: install test figures run clean

install:
	pip install -e .[dev]

test:
	PYTHONPATH=src pytest

figures:
	PYTHONPATH=src python scripts/make_all_figures.py

run:
	PYTHONPATH=src python scripts/run_pipeline.py

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
