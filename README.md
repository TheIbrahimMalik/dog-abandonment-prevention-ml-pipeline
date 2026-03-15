# Dog Abandonment Prevention ML Pipeline

**Author:** Ibrahim Malik - initial version Sep 2025, refactored and extended 2026

A production-style machine learning pipeline that turns animal shelter intake/outcome records into decision support for reducing dog returns after adoption.

The system is designed around three objectives:

1. **Risk prediction (binary):** predict the probability a dog is returned to the shelter within a chosen time horizon after adoption (e.g. 90 days).
2. **Reason classification (multi-class):** for returned dogs, predict the most likely return reason proxy derived from the next intake event.
3. **Intervention guidance (rule-based):** map predicted risk and reason to recommended actions such as follow-ups, training support, veterinary checks, and adopter guidance.

This repository is a **retrainable template**: the pipeline is reusable, but models should be retrained on local shelter data to generalise reliably.

---

## Proposal and Technical Report

- Design document: `reports/design_doc.md`
- Technical report: `reports/technical_report.md`

---

## Architecture

```
Intakes + Outcomes (raw CSV)
        ↓
Preprocess (clean + join + adoption episodes)
        ↓
Label builder (adoption → next intake)
        ↓
Binary risk model (return within N days)
        ↓
Reason model (reason proxy for returns)
        ↓
Decision layer (intervention recommendations)
```

---

## Dataset

This project uses the City of Austin Open Data Portal (Public Domain):

1. **Austin Animal Center Intakes** (10/01/2013 to 05/05/2025)
   https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Intakes-10-01-2013-to-05-05-2/wter-evkm/about_data

2. **Austin Animal Center Outcomes** (10/01/2013 to 05/05/2025)
   https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes-10-01-2013-to-05-05-/9t4d-g238/about_data

Download both datasets as CSV and place them in:

```
data/raw/aac_intakes.csv
data/raw/aac_outcomes.csv
```

---

## Repository Structure

```
.
├── src/
│   └── dog_abandonment/
│       ├── __init__.py
│       ├── preprocessing.py      # Data cleaning, joining, adoption episode construction
│       ├── return_labels.py      # Label builder (adoption → next intake)
│       ├── train.py              # Model training (binary risk + reason classifier)
│       └── cli.py                # CLI entrypoints (preprocess, train)
├── reports/
│   ├── design_doc.md
│   ├── technical_report.md
│   └── metrics/
├── figures/
├── data/
│   ├── raw/                      # not committed
│   └── processed/                # not committed
├── artifacts/
│   ├── metrics/                  # safe to commit
│   └── models/                   # not committed
├── legacy/                       # Original submission artefacts (provenance)
├── .gitignore
├── CITATION.cff
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## How to Run

### View-only (recommended for recruiters)

- Read `reports/design_doc.md` and `reports/technical_report.md`
- Review the pipeline source code under `src/dog_abandonment/`
- Review plots in `figures/` and metric outputs in `artifacts/metrics/`

---

### Full Execution (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# 1) Generate modelling dataset
dog-abandonment preprocess

# 2) Train models (default horizon: 90 days)
dog-abandonment train --horizon 90
```

### Predict (example)

```bash
mkdir -p outputs
dog-abandonment predict --input examples/sample_input.csv --out outputs/predictions_sample.csv
```


### Predict (example)

```bash
mkdir -p outputs
dog-abandonment predict --input examples/sample_input.csv --out outputs/predictions_sample.csv
```


**Outputs:**

- `data/processed/return_risk_dataset.csv`  - generated, not committed
- `artifacts/models/*.joblib` - generated, not committed
- `artifacts/metrics/*.json` - generated, safe to commit

---

## Current Baseline Results

Evaluated on a time-based train/test split with `return_within_90d` as the target:

- **Binary risk model PR-AUC (test):** ~0.12 (dummy baseline ~0.09)
- **Reason model:** highly imbalanced; baseline is close to predicting the most frequent reason

Results will vary by horizon, feature set, and train/test split time window.

---

## ML Engineering Concepts Demonstrated

- Time-aware preprocessing and evaluation to reduce leakage
- Episode and label construction from event streams (adoption → next intake)
- Robust datetime parsing including time zones and DST handling
- Reproducible CLI workflow via `preprocess` and `train`
- GitHub-safe artefact handling - no raw data or model binaries committed

---

## Attribution

Originally completed as part of the Udacity AWS Machine Learning Engineer Nanodegree Capstone (2025). Refactored and extended in 2026 for professional portfolio presentation.

---

## Licence

[MIT License](LICENSE)