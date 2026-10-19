# Ofsted decline-risk project (NCHNAP691)

Predicting which Good schools are at risk of decline at their next Ofsted
inspection, using public Ofsted and DfE data.

## Set-up

Place the downloads listed in `data/manifest.csv` under `data/raw/`, with
the absence ZIP extracted into `data/raw/absence/` and the GIAS ZIP into
`data/raw/gias/`. Then, from the repository root:

```
pip install -e ".[dev]"
```

```
pytest
```

```
python scripts/build_label_table.py
```

The last command writes the label tables and `data/interim/data_quality.json`.
