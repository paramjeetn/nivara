PYTHON ?= python
PYTEST ?= pytest
STREAMLIT ?= streamlit
PIP ?= pip

ifeq ($(OS),Windows_NT)
    PYTHON ?= python.bat
    PYTEST ?= pytest.bat
    STREAMLIT ?= streamlit.bat
    PIP ?= pip.bat
endif

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
