# Quickstart (local)

## 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 2) Download data
Place the City of Austin datasets in:
* `data/raw/aac_intakes.csv`
* `data/raw/aac_outcomes.csv`

## 3) Preprocess + train
```bash
dog-abandonment preprocess
dog-abandonment train --horizon 90
```

## 4) Predict (example)
```bash
mkdir -p outputs
dog-abandonment predict --input examples/sample_input.csv --out outputs/predictions_sample.csv
```
