VENV_PYTHON := $(wildcard .venv/bin/python)
PYTHON ?= $(if $(VENV_PYTHON),$(VENV_PYTHON),python3)
PYTHONPATH := src
DATASET ?= cicids2017
MAX_ROWS ?=
export PYTHONPATH

.PHONY: install doctor data train quick-train explain evaluate pipeline unified all summary wazuh api test clean

DATASET_ARG := --dataset $(DATASET)
MAX_ROWS_ARG := $(if $(MAX_ROWS),--max-rows $(MAX_ROWS),)

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

doctor:
	@$(PYTHON) -c "import sys; print('Python:', sys.executable)"
	@$(PYTHON) -c "import yaml, seaborn, pyarrow, sklearn, xgboost, torch; print('Dependencies: OK')"

data:
	$(PYTHON) -m soc_ready_ids.data.preprocessor --config config.yaml $(DATASET_ARG) $(MAX_ROWS_ARG)
	$(PYTHON) -m soc_ready_ids.data.feature_selector --config config.yaml $(DATASET_ARG)

train:
	$(PYTHON) -m soc_ready_ids.models.train_all --config config.yaml $(DATASET_ARG)

quick-train:
	$(PYTHON) -m soc_ready_ids.models.train_all --config config.yaml $(DATASET_ARG) --models random_forest

explain:
	$(PYTHON) -m soc_ready_ids.explainability.shap_explainer --config config.yaml $(DATASET_ARG)
	$(PYTHON) -m soc_ready_ids.explainability.lime_explainer --config config.yaml $(DATASET_ARG)

evaluate:
	$(PYTHON) -m soc_ready_ids.evaluation.run_all --config config.yaml $(DATASET_ARG)

pipeline: data train evaluate

unified:
	$(MAKE) pipeline DATASET=combined PYTHON=$(PYTHON)

all:
	$(MAKE) pipeline DATASET=cicids2017 PYTHON=$(PYTHON)
	$(MAKE) pipeline DATASET=bot-iot PYTHON=$(PYTHON)
	$(MAKE) pipeline DATASET=combined PYTHON=$(PYTHON)
	$(MAKE) summary PYTHON=$(PYTHON)

summary:
	$(PYTHON) -m soc_ready_ids.evaluation.dataset_summary --config config.yaml

wazuh:
	$(PYTHON) wazuh/validate_integration.py --project-root . $(DATASET_ARG)

api:
	$(PYTHON) -m soc_ready_ids.api.app $(DATASET_ARG)

test:
	$(PYTHON) -m pytest --cov=soc_ready_ids --cov-report=term-missing --cov-report=html --cov-fail-under=95

clean:
	rm -rf .pytest_cache htmlcov .coverage .coverage.*
	find . -path './.venv' -prune -o -type d -name '__pycache__' -exec rm -rf {} +
	find . -path './.venv' -prune -o -type f -name '.DS_Store' -delete
