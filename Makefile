.PHONY: data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON_INTERPRETER = "python3"

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/process_seer.py

## Delete all compiled Python files and processed datasets
clean: 
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find data/processed -type f -name "*.csv" -delete