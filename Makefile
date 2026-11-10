PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
STREAMLIT ?= $(PYTHON) -m streamlit

export PYTHONPATH := src

.PHONY: setup run demo eval realdata test

setup:
	$(PIP) install -e ".[dev]"

run:
	$(STREAMLIT) run src/campus_energy/dashboard/app.py

demo:
	$(STREAMLIT) run src/campus_energy/dashboard/app.py -- --demo

eval:
	$(PYTHON) -m campus_energy.eval.run_all

realdata:
	$(PYTHON) scripts/run_realdata_eval.py

test:
	$(PYTEST) tests/ -v
