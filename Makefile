.PHONY: setup test lint quick full eda validate audit

setup:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check --no-cache src scripts tests

quick:
	python scripts/run_quick_pipeline.py

full:
	python scripts/run_full_pipeline.py

eda:
	python scripts/run_eda.py --config configs/quick.yaml

validate:
	python scripts/validate_data.py --config configs/full.yaml

audit:
	python scripts/audit_raw_data.py

