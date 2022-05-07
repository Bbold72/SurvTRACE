.PHONY: clean data run experiments results

#################################################################################
# GLOBALS                                                                       #
#################################################################################
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
run: clean data run experiments results

data: seer dataset

seer:
	@$(PYTHON_INTERPRETER) src/data/process_seer.py
	@echo ">>> Creating Datasets with $(NUM_RUNS) runs."

dataset:
	@echo ">>> Creating Datasets with $(NUM_RUNS) runs."
	@$(PYTHON_INTERPRETER) src/data/make_datasets.py --num_runs=$(NUM_RUNS)

experiments:
	@echo ">>> Running experiments with $(NUM_RUNS) runs."
	@$(PYTHON_INTERPRETER) src/experiments/make_experiments.py --num_runs=$(NUM_RUNS)

result:
	@echo ">>> Printing results."
	@$(PYTHON_INTERPRETER) src/results/make_results.py

## Delete all compiled Python files and processed datasets
clean:
	@echo ">>> Cleaning files."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@find data/processed ! -name '.gitkeep' -type f -exec rm -f {} +