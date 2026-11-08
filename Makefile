PY ?= python

.PHONY: install test lint data reproduce full validate real-histdata real-bi5 advisor
install:
	$(PY) -m pip install -e ../skeptic -e ".[dev]"
test:
	$(PY) -m pytest
lint:
	ruff check src tests studies
	mypy src
data:                 ## real Dukascopy one-minute bid/ask bars, 2019 and 2024-2026 (~100 MB)
	$(PY) -m atlas.download --m1
reproduce:            ## checked-in 6-month AUDUSD sample (real data): heatmap, event study, calibration (~2 min)
	$(PY) studies/run_atlas.py --quick
full:                 ## the study behind the README, on real data (needs `make data`, ~10 min)
	$(PY) studies/run_atlas.py
validate:             ## pipeline validation: planted effects in synthetic quotes (~15 min)
	$(PY) studies/run_atlas.py --source synthetic
real-histdata:        ## alternative source: HistData tick zips in data/raw
	$(PY) studies/run_atlas.py --source histdata --raw data/raw
real-bi5:             ## alternative source: Dukascopy hourly tick files fetched with atlas-download
	$(PY) studies/run_atlas.py --source bi5 --raw data/raw
advisor:
	$(PY) -m atlas.advisor --pair AUDUSD --notional 5e6 --urgency 2 --start 2026-11-02
