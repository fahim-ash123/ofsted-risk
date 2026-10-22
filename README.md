# What the Previous Report Cannot Tell

Predicting which schools judged Good at their last graded Ofsted inspection are at risk of decline at their next inspection, using only information available before that inspection. This repository holds the code for my summative assessment in Implementing Data Science (NCHNAP691), part of my BSc Data Science at NU London.

The project compares basic classifiers on school context (logistic regression and gradient boosting) with an NLP model on the text of the previous inspection report (sentence embeddings with logistic regression, plus a TF-IDF baseline), and a late fusion of the two. Models are trained on 2021/22 and 2022/23 and tested once on 2023/24.

**Headline result:** school context ranks schools at about twice the precision of random selection (PR-AUC 0.299 against a base rate of 0.141), while the previous report's improvement section predicts decline only marginally better than chance and adds nothing once context is known.

## Repository layout

| Folder | Contents |
|---|---|
| `src/ofsted_risk/` | The pipeline package: settings, loading, labels and checks, report download and parsing, text extraction, feature joins, embeddings, models and evaluation |
| `scripts/` | One script per stage, run in order (see below) |
| `tests/` | 43 test functions, run by pytest as 56 test cases |
| `data/manifest.csv` | Source, download address and download date of every raw file |

No data is committed. Every raw file is public and listed in the manifest. The downloaded inspection reports contain the names of headteachers, governors and inspectors, so only the cleaned improvement sections, with names and page headers removed, are used in the analysis.

## Running the pipeline

Install the pinned packages and run the tests:

    pip install -r requirements.txt
    pytest

Place the raw files listed in `data/manifest.csv` under `data/raw/`, with the absence ZIP extracted into `data/raw/absence/` and the GIAS ZIP into `data/raw/gias/`. Then, from the repository root, run each stage in order:

    PYTHONPATH=src python scripts/build_label_table.py
    PYTHONPATH=src python scripts/download_reports.py
    PYTHONPATH=src python scripts/build_text_table.py
    PYTHONPATH=src python scripts/build_model_table.py
    PYTHONPATH=src python scripts/run_modelling.py

The report download makes two requests per school, one second apart, so it takes several hours and resumes from where it stopped if interrupted. The modelling stage writes every table and figure to `data/processed/results/`.

The accompanying Deepnote notebook runs the same stages, shows every output, and reruns the analysis from the submitted dataset files.

## Environment

Developed and run on Deepnote with Python 3.13. Package versions are pinned in `requirements.txt`.

## Data licence

All source data is published by Ofsted and the Department for Education under the Open Government Licence v3.0.
