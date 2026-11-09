PYTHON ?= python.bat
PYTEST ?= pytest.bat
STREAMLIT ?= streamlit.bat

.PHONY: eval test run realdata

eval:
	$(PYTHON) -m campus_energy.eval.run_all

test:
	$(PYTEST) tests/ -v

run:
	$(STREAMLIT) run src/campus_energy/dashboard/app.py

realdata:
	$(PYTHON) scripts/run_realdata_eval.py
