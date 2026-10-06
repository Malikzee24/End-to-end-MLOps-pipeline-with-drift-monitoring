.PHONY: install test lint demo train serve monitor traffic up down clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=mlops_pipeline --cov-report=term-missing

lint:
	ruff check src tests scripts

train:            ## train + register + promote the first champion
	mlops train

serve:            ## inference API on :8000
	mlops serve

monitor:          ## one monitoring cycle (ARGS=--loop for continuous)
	mlops monitor $(ARGS)

traffic:          ## synthetic traffic with drift against a local API
	python scripts/traffic_generator.py --api-url http://localhost:8000

demo:             ## 22 simulated weeks: drift -> alert -> retrain -> promote (~1-2 min)
	mlops demo --plot docs/images/demo_timeline.png
	cp state/demo/reports/latest.html docs/sample_drift_report.html

up:
	docker compose --profile demo up --build

down:
	docker compose --profile demo down -v

clean:
	rm -rf state mlruns .pytest_cache .ruff_cache .coverage
