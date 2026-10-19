"""Download the previous inspection report text for every labelled school.

Run from the repository root after ``build_label_table.py``. Start with a
trial run to check the matching works:

    python scripts/download_reports.py --limit 5

Then run without ``--limit``. Records are written to the machine's local
disk and saved to ``data/interim/reports/`` as a new shard after every
batch, because the project drive is slow for many small writes. The script
can be stopped at any time and started again, including after a machine
restart; it restores saved shards and carries on from where it stopped.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ofsted_risk import config, reports


def parse_args() -> argparse.Namespace:
    """Read command-line options.

    Returns:
        Parsed arguments with ``label_table``, ``shard_dir``, ``local``,
        ``delay``, ``batch`` and ``limit``.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label-table", type=Path, default=Path("data/interim/label_table.csv"))
    parser.add_argument("--shard-dir", type=Path, default=Path("data/interim/reports"))
    parser.add_argument("--local", type=Path, default=Path("/tmp/ofsted_reports/reports.jsonl"))
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def count_lines(path: Path) -> int:
    """Count non-empty lines in a results file.

    Args:
        path: JSON Lines file, which may not exist yet.

    Returns:
        The number of records in the file.
    """
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> None:
    """Download pending reports in batches, saving a shard after each batch."""
    args = parse_args()
    table = pd.read_csv(args.label_table, dtype=str, keep_default_na=False)
    targets = reports.report_targets([table], first_year=config.TEST_YEAR)

    restored = reports.restore_from_shards(args.shard_dir, args.local)
    saved = sum(count_lines(shard) for shard in args.shard_dir.glob("part_*.jsonl")) if args.shard_dir.exists() else 0
    print(f"{len(targets)} reports needed, {restored} already recorded", flush=True)

    remaining = args.limit
    totals: dict = {}
    try:
        while remaining is None or remaining > 0:
            size = args.batch if remaining is None else min(args.batch, remaining)
            counts = reports.download_reports(targets, args.local, delay=args.delay, limit=size)
            if not counts:
                break
            for status, number in counts.items():
                totals[status] = totals.get(status, 0) + number
            if remaining is not None:
                remaining -= sum(counts.values())
            reports.save_shard(args.local, saved, args.shard_dir)
            saved = count_lines(args.local)
            print(f"Saved to project folder: {saved} of {len(targets)} recorded, this run {totals}", flush=True)
    finally:
        # Covers a stop part-way through a batch, so finished reports are kept.
        reports.save_shard(args.local, saved, args.shard_dir)

    print(f"This run: {totals}")
    if args.limit:
        lines = args.local.read_text(encoding="utf-8").splitlines()
        for line in lines[-args.limit:]:
            record = json.loads(line)
            record.pop("pages", None)
            print(record)


if __name__ == "__main__":
    main()
