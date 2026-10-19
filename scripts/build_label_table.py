"""Build the labelled population and the data-quality report.

Run from the repository root:

    python scripts/build_label_table.py

The script writes ``data/interim/label_table.csv`` for the primary
population, ``data/interim/label_table_with_predecessors.csv`` for the
sensitivity run, and ``data/interim/data_quality.json`` holding every
figure quoted in the report's data-understanding section.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ofsted_risk import checks, config, ingest, labels


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments with ``raw_dir`` and ``out_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    return parser.parse_args()


def main() -> None:
    """Build the label tables and write the data-quality report."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    absence = ingest.load_absence(args.raw_dir)
    gias = ingest.load_gias(args.raw_dir)
    report = {"years": {}}
    primary_tables, sensitivity_tables = [], []

    for year in config.TARGET_YEARS:
        inspections = ingest.load_between(year, args.raw_dir)
        snapshot = ingest.load_as_at(config.STANDING_SNAPSHOT[year], args.raw_dir)
        later = ingest.load_as_at(config.COMPLETENESS_SNAPSHOT[year], args.raw_dir)
        census = ingest.load_census(year, args.raw_dir)

        table = labels.build_label_table(year, inspections, snapshot)
        wider = labels.build_label_table(year, inspections, snapshot, include_predecessor=True)
        primary_tables.append(table)
        sensitivity_tables.append(wider)

        year_absence = absence[absence["time_period"] == config.ABSENCE_PERIOD[year]]
        report["years"][year] = {
            "source_rows": int(len(inspections)),
            "blank_rows_dropped": inspections.attrs["blank_rows_dropped"],
            "label_steps": table.attrs,
            "class_balance": checks.class_balance(table),
            "class_balance_with_predecessors": checks.class_balance(wider),
            "previous_grade_agreement": checks.previous_grade_agreement(inspections, snapshot),
            "late_publications": checks.late_publications(inspections, later, config.YEAR_WINDOWS[year]),
            "coverage_absence": checks.join_coverage(table, year_absence),
            "coverage_census": checks.join_coverage(table, census),
            "coverage_gias": checks.join_coverage(table, gias),
            "census_source": census.attrs["source"],
            "census_non_numeric": census.attrs["non_numeric"],
            "sixth_form_non_grade_codes": ingest.non_grade_codes(
                inspections["sixth_form_provision_where_applicable"]
            ),
            "prior_ungraded_outcome": table["prior_ungraded_outcome"].value_counts().to_dict(),
        }

    pd.concat(primary_tables).to_csv(args.out_dir / "label_table.csv", index=False)
    pd.concat(sensitivity_tables).to_csv(args.out_dir / "label_table_with_predecessors.csv", index=False)
    (args.out_dir / "data_quality.json").write_text(json.dumps(report, indent=2))

    summary = pd.DataFrame({year: values["class_balance"] for year, values in report["years"].items()}).T
    print(summary.to_string())


if __name__ == "__main__":
    main()
