"""Data-quality checks that feed the Real-world Data Understanding section.

Each check returns plain numbers rather than printing, so the same results
can be written to the data-quality report, asserted in tests and quoted in
the written report without being retyped.
"""

import pandas as pd

from ofsted_risk.ingest import parse_grade, parse_ofsted_date
from ofsted_risk.labels import RELATES


def previous_grade_agreement(inspections: pd.DataFrame, snapshot: pd.DataFrame) -> dict:
    """Compare each inspection's stated previous grade with the prior snapshot.

    Agreement shows that "previous overall effectiveness" in the year file
    means the previous graded inspection, and that the two sources are
    consistent enough to rely on one of them.

    Args:
        inspections: One year of inspections from ``load_between``.
        snapshot: The "as at" file for 31 August before that year.

    Returns:
        A mapping with the number of rows compared, the share whose
        previous grade matches the snapshot's latest graded overall
        effectiveness, and the share whose previous inspection number
        matches the snapshot's latest graded inspection number.
    """
    reference = snapshot[["urn", "overall_effectiveness", "inspection_number_of_latest_graded_inspection"]].rename(
        columns={"overall_effectiveness": "overall_effectiveness_snapshot"}
    )
    merged = inspections.merge(reference, on="urn", how="inner")
    same_school = merged[merged[RELATES].eq("Yes").fillna(False)]
    grade_match = parse_grade(same_school["previous_overall_effectiveness"]).eq(
        parse_grade(same_school["overall_effectiveness_snapshot"])
    )
    number_match = (
        merged["previous_inspection_number"].eq(merged["inspection_number_of_latest_graded_inspection"]).fillna(False)
    )
    return {
        "rows_compared": int(len(same_school)),
        "grade_agreement": round(float(grade_match.mean()), 4),
        "inspection_number_agreement": round(float(number_match.mean()), 4),
    }


def late_publications(inspections: pd.DataFrame, later_snapshot: pd.DataFrame, window: tuple) -> dict:
    """Find inspections in a year that the year file missed but a later snapshot holds.

    The year file only includes reports published by a cut-off date. A
    snapshot published a year later lists each school's latest inspection,
    so inspections from the window that appear there but not in the year
    file were published late. This is a lower bound, because a school
    inspected again in the meantime no longer shows the earlier inspection.

    Args:
        inspections: One year of inspections from ``load_between``.
        later_snapshot: An "as at" file published after the year ended.
        window: Start and end dates of the academic year as ISO strings.

    Returns:
        A mapping with counts of graded and ungraded inspections found in
        the window, how many of each are missing from the year file, and
        the overall effectiveness of the missing graded inspections.
    """
    known = set(inspections["inspection_number"])
    start, end = (pd.Timestamp(date) for date in window)

    graded_date = parse_ofsted_date(later_snapshot["inspection_start_date"])
    graded = later_snapshot[graded_date.between(start, end)]
    graded_missing = graded[~graded["inspection_number_of_latest_graded_inspection"].isin(known)]

    ungraded_date = parse_ofsted_date(later_snapshot["date_of_latest_ungraded_inspection"])
    ungraded = later_snapshot[ungraded_date.between(start, end)]
    ungraded_missing = ungraded[
        ~ungraded["latest_ungraded_inspection_number_since_last_graded_inspection"].isin(known)
    ]

    return {
        "graded_in_window": int(len(graded)),
        "graded_missing": int(len(graded_missing)),
        "graded_missing_outcomes": {
            str(grade): int(count)
            for grade, count in parse_grade(graded_missing["overall_effectiveness"]).value_counts().sort_index().items()
        },
        "graded_in_window_below_good_share": round(
            float(parse_grade(graded["overall_effectiveness"]).isin([3, 4]).mean()), 3
        ),
        "ungraded_in_window": int(len(ungraded)),
        "ungraded_missing": int(len(ungraded_missing)),
    }


def join_coverage(label_table: pd.DataFrame, other: pd.DataFrame, key: str = "urn") -> dict:
    """Measure how many labelled schools find a match in another source.

    Args:
        label_table: Output of ``build_label_table`` for one year.
        other: The source to join, already filtered to the right year.
        key: Join column present in both frames.

    Returns:
        A mapping with the match share and the phases of unmatched schools.
    """
    matched = label_table[key].isin(set(other[key]))
    return {
        "match_share": round(float(matched.mean()), 4),
        "unmatched_by_phase": label_table.loc[~matched, "ofsted_phase"].value_counts().to_dict(),
    }


def class_balance(label_table: pd.DataFrame) -> dict:
    """Summarise population size and label rates for one year.

    Args:
        label_table: Output of ``build_label_table`` for one year.

    Returns:
        A mapping with counts and rates for both labels.
    """
    graded = label_table[label_table["is_graded"]]
    return {
        "population": int(len(label_table)),
        "primary_positive": int(label_table["label_primary"].sum()),
        "primary_rate": round(float(label_table["label_primary"].mean()), 4),
        "graded": int(len(graded)),
        "strict_positive": int(graded["label_strict"].sum()),
        "strict_rate": round(float(graded["label_strict"].mean()), 4),
        "ungraded": int((~label_table["is_graded"]).sum()),
    }
