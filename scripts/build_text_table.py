"""Extract the cleaned improvement section from every downloaded report.

Run from the repository root after ``download_reports.py`` has finished:

    python scripts/build_text_table.py

The script writes ``improvement_text.csv``, one row per report, and
``text_quality.json`` with the figures quoted in the report: how the
section was found, how long it is by year, and how many names were removed.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ofsted_risk import text


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments with ``shard_dir`` and ``out_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard-dir", type=Path, default=Path("data/interim/reports"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    return parser.parse_args()


def latest_records(shard_dir: Path) -> dict:
    """Read every shard, keeping the latest record for each report.

    Args:
        shard_dir: Folder holding ``part_*.jsonl`` shards.

    Returns:
        A mapping from report key to its latest record.
    """
    records = {}
    for shard in sorted(shard_dir.glob("part_*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                records[record["report_key"]] = record
    return records


def main() -> None:
    """Build the text table and write it with its quality summary."""
    args = parse_args()
    table = text.build_text_table(latest_records(args.shard_dir).values())
    table["report_year"] = table["listed_date"].map(text.report_year)
    downloaded = table[table["status"] == "ok"]

    quality = {
        "reports": int(len(table)),
        "status": table["status"].value_counts().to_dict(),
        "end_marker": downloaded["end_marker"].value_counts().to_dict(),
        "capped_by_year": downloaded.groupby("report_year")["capped"].sum().astype(int).to_dict(),
        "median_characters_by_year": downloaded.groupby("report_year")["characters"].median().astype(int).to_dict(),
        "names_removed_total": int(downloaded["names_removed"].sum()),
        "reports_with_names_removed": int((downloaded["names_removed"] > 0).sum()),
        "no_heading_examples": downloaded.loc[downloaded["end_marker"] == "no_heading", "report_key"].head(10).tolist(),
        "no_match_reports": table.loc[table["status"] == "no_match", "report_key"].tolist(),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.drop(columns="report_year").to_csv(args.out_dir / "improvement_text.csv", index=False)
    (args.out_dir / "text_quality.json").write_text(json.dumps(quality, indent=2))

    for key in ("reports", "status", "end_marker", "capped_by_year", "median_characters_by_year"):
        print(f"{key}: {quality[key]}")
    print(f"names removed: {quality['names_removed_total']} in {quality['reports_with_names_removed']} reports")
    print(f"no heading, first examples: {quality['no_heading_examples']}")
    print(f"no match: {quality['no_match_reports']}")


if __name__ == "__main__":
    main()
