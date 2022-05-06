.PHONY: data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = "python3"
NUM_RUNS = 10

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
make: clean run

data: seer dataset

seer:
	@$(PYTHON_INTERPRETER) src/data/process_seer.py
	@echo ">>> Creating Datasets with $(NUM_RUNS) runs."

dataset:
	@echo ">>> Creating Datasets with $(NUM_RUNS) runs."
	@$(PYTHON_INTERPRETER) src/data/make_datasets.py --runs=$(NUM_RUNS)

## Delete all compiled Python files and processed datasets
clean:
	@echo ">>> Cleaning files."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find data/processed ! -name '.gitkeep' -type f -exec rm -f {} +