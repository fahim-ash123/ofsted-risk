"""Build the modelling table: one row per labelled inspection with features and text.

Run from the repository root after ``build_text_table.py``:

    python scripts/build_model_table.py --raw-dir data/raw --interim-dir data/interim

It writes ``data/processed/model_table.csv``, the dataset submitted with
the report, and prints its size, label rates, text coverage and the share of
missing values in each feature by year.
"""

import argparse
from pathlib import Path

import pandas as pd

from ofsted_risk import config, features, ingest


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments with ``raw_dir``, ``interim_dir`` and ``out_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--interim-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    return parser.parse_args()


def main() -> None:
    """Join the sources, check the result and write the modelling table."""
    args = parse_args()
    labels = pd.read_csv(args.interim_dir / "label_table.csv", dtype=str, keep_default_na=False)
    texts = pd.read_csv(args.interim_dir / "improvement_text.csv", dtype={"report_key": str})
    census = {year: ingest.load_census(year, args.raw_dir) for year in config.TARGET_YEARS}
    table = features.build_model_table(
        labels, texts, census, ingest.load_absence(args.raw_dir), ingest.load_gias(args.raw_dir)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "model_table.csv", index=False)

    print(f"rows: {len(table)}, columns: {table.shape[1]}")
    summary = table.groupby("year").agg(
        rows=("urn", "size"),
        at_risk_rate=("label_primary", "mean"),
        with_text=("improvement_text", lambda column: column.notna().mean()),
    )
    print(summary.round(4).to_string())
    missing = features.missing_by_year(table)
    print("Share missing by year (features with any missing values):")
    print(missing.loc[:, missing.gt(0).any()].to_string())
    size = (args.out_dir / "model_table.csv").stat().st_size
    print(f"model_table.csv: {size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
