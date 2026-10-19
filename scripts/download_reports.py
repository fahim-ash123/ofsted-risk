"""Download the previous inspection report text for every labelled school.

Run from the repository root after ``build_label_table.py``. Start with a
trial run to check the matching works:

    python scripts/download_reports.py --limit 5

Then run without ``--limit``. The script can be stopped at any time and
started again; it carries on from where it stopped.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ofsted_risk import config, reports


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments with ``label_table``, ``output``, ``delay`` and
        ``limit``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-table", type=Path, default=Path("data/interim/label_table.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/reports.jsonl"))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Download pending reports and print a summary of the run."""
    args = parse_args()
    table = pd.read_csv(args.label_table, dtype=str, keep_default_na=False)
    targets = reports.report_targets([table], first_year=config.TEST_YEAR)
    already = len(reports.completed_keys(args.output))
    print(f"{len(targets)} reports needed, {already} already recorded", flush=True)

    counts = reports.download_reports(targets, args.output, delay=args.delay, limit=args.limit)
    print(f"This run: {counts}")

    if args.limit:
        with args.output.open(encoding="utf-8") as handle:
            for line in list(handle)[-args.limit:]:
                record = json.loads(line)
                record.pop("pages", None)
                print(record)


if __name__ == "__main__":
    main()
