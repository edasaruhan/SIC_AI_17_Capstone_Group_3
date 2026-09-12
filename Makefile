.PHONY: setup setup-core setup-locked test coverage lint check quick full baseline verify-baseline validate-sprint2 eda validate audit final-eval verify-final validate-final final-docs final-screenshots

setup:
	python -m pip install -e ".[graph,app,dev]"

setup-core:
	python -m pip install -e ".[dev]"

setup-locked:
	python -m pip install -c constraints/py312-dev.txt -e ".[graph,app,dev]"

test:
	python -m pytest

coverage:
	python -m pytest --cov=argus --cov-report=term-missing --cov-fail-under=76

lint:
	python -m ruff check --no-cache src scripts tests app.py
	python -m ruff format --no-cache --check src scripts tests app.py

check: test lint
	python -m pip check

quick:
	python scripts/run_quick_pipeline.py

full:
	python scripts/run_full_pipeline.py

baseline:
	python scripts/train_baselines.py --config configs/baseline.yaml

verify-baseline:
	python scripts/verify_baselines.py --config configs/baseline.yaml

validate-sprint2:
	python scripts/validate_sprint2.py --config configs/baseline.yaml

eda:
	python scripts/run_eda.py --config configs/quick.yaml

validate:
	python scripts/validate_data.py --config configs/full.yaml

audit:
	python scripts/audit_raw_data.py

final-eval:
	python scripts/run_final_evaluation.py --config configs/final_evaluation.yaml

verify-final:
	python scripts/verify_final_evaluation.py --config configs/final_evaluation.yaml

validate-final:
	python scripts/validate_final_evaluation.py --config configs/final_evaluation.yaml

final-docs:
	python scripts/update_final_documentation.py --config configs/final_evaluation.yaml

final-screenshots:
	python scripts/capture_final_streamlit_screenshots.py --artifact-root artifacts/sprint5
