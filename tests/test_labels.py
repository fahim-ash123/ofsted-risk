"""Tests for the eligibility rules, labels and snapshot joins."""

import pandas as pd
import pytest

from conftest import make_inspection, make_snapshot_row
from ofsted_risk.labels import (
    OUTCOME,
    RELATES,
    assign_labels,
    attach_prior_ungraded,
    build_label_table,
    eligible,
    first_per_school,
    prior_ungraded_context,
    repair_masked_previous_numbers,
)


def test_eligible_keeps_only_previously_good_schools_in_scope(frame_of):
    """Each exclusion rule removes exactly the row it targets."""
    frame = frame_of(
        make_inspection(urn="1"),
        make_inspection(urn="2", previous_overall_effectiveness="3"),
        make_inspection(urn="3", **{RELATES: "No"}),
        make_inspection(urn="4", ofsted_phase="Nursery"),
        make_inspection(urn="5", inspection_type_grouping="Requires Improvement monitoring Visit"),
    )

    assert eligible(frame)["urn"].tolist() == ["1"]
    assert eligible(frame, include_predecessor=True)["urn"].tolist() == ["1", "3"]


@pytest.mark.parametrize(
    ("overall", "outcome", "primary", "strict"),
    [
        ("3", pd.NA, 1, 1),
        ("4", pd.NA, 1, 1),
        ("2", pd.NA, 0, 0),
        ("1", pd.NA, 0, 0),
        ("9", "School remains Good (Concerns) - S5 Next", 1, pd.NA),
        ("9", "School remains Good", 0, pd.NA),
        ("9", "School remains Good (Improving) - S5 Next", 0, pd.NA),
    ],
)
def test_assign_labels_follows_the_label_definition(frame_of, overall, outcome, primary, strict):
    """Graded and ungraded outcomes map to the agreed labels.

    Args:
        frame_of: Frame builder fixture.
        overall: Overall effectiveness code.
        outcome: Ungraded outcome text.
        primary: Expected primary label.
        strict: Expected strict label, missing for ungraded inspections.
    """
    frame = frame_of(make_inspection(overall_effectiveness=overall, **{OUTCOME: outcome}))

    labelled = assign_labels(frame)

    assert labelled.loc[0, "label_primary"] == primary
    if pd.isna(strict):
        assert pd.isna(labelled.loc[0, "label_strict"])
    else:
        assert labelled.loc[0, "label_strict"] == strict


def test_assign_labels_refuses_unseen_outcomes(frame_of):
    """A new outcome wording must stop the build rather than be dropped."""
    frame = frame_of(make_inspection(**{OUTCOME: "Standards maintained"}))

    with pytest.raises(ValueError, match="Standards maintained"):
        assign_labels(frame)


def test_first_per_school_keeps_the_earliest_visit(frame_of):
    """A Concerns visit followed by a graded inspection keeps the visit."""
    frame = frame_of(
        make_inspection(inspection_number="B", inspection_start_date="04/06/2024", overall_effectiveness="2", **{OUTCOME: pd.NA}),
        make_inspection(inspection_number="A", inspection_start_date="21/11/2023", **{OUTCOME: "School remains Good (Concerns) - S5 Next"}),
    )

    kept = first_per_school(assign_labels(frame))

    assert kept["inspection_number"].tolist() == ["A"]
    assert kept.attrs["repeat_inspections_dropped"] == 1


def test_repair_restores_masked_numbers_only_when_dates_match(frame_of):
    """A masked number is restored from the snapshot only for the same inspection."""
    frame = frame_of(
        make_inspection(urn="1", previous_inspection_number="########"),
        make_inspection(urn="2", previous_inspection_number="####", previous_inspection_start_date="01/01/2015"),
    )
    snapshot = frame_of(
        make_snapshot_row(urn="1", inspection_number_of_latest_graded_inspection="ITS111"),
        make_snapshot_row(urn="2", inspection_number_of_latest_graded_inspection="ITS222"),
    )

    repaired = repair_masked_previous_numbers(frame, snapshot)

    assert repaired.loc[0, "previous_inspection_number"] == "ITS111"
    assert pd.isna(repaired.loc[1, "previous_inspection_number"])
    assert repaired.attrs["masked_previous_numbers"] == {"masked": 2, "repaired": 1, "unresolved": 1}


def test_prior_ungraded_context_separates_none_from_not_recorded(frame_of):
    """No ungraded visit and an ungraded visit with no recorded outcome differ."""
    snapshot = frame_of(
        make_snapshot_row(urn="1"),
        make_snapshot_row(urn="2", number_of_ungraded_inspections_since_the_last_graded_inspection="1"),
        make_snapshot_row(
            urn="3",
            number_of_ungraded_inspections_since_the_last_graded_inspection="1",
            ungraded_inspection_overall_outcome="School remains Good",
        ),
    )

    context = prior_ungraded_context(snapshot)

    assert context["prior_ungraded_outcome"].tolist() == ["none", "not_recorded", "School remains Good"]


def test_attach_prior_ungraded_ignores_history_from_another_cycle(frame_of):
    """Snapshot history is only used when it describes the same graded inspection."""
    frame = frame_of(make_inspection(urn="1"), make_inspection(urn="2"))
    snapshot = frame_of(
        make_snapshot_row(urn="1", ungraded_inspection_overall_outcome="School remains Good",
                          number_of_ungraded_inspections_since_the_last_graded_inspection="1"),
        make_snapshot_row(urn="2", inspection_number_of_latest_graded_inspection="DIFFERENT"),
    )

    attached = attach_prior_ungraded(frame, prior_ungraded_context(snapshot))

    assert attached["prior_ungraded_outcome"].tolist() == ["School remains Good", "unknown"]
    assert attached.attrs["prior_context_unknown"] == 1


def test_build_label_table_returns_one_labelled_row_per_school(frame_of):
    """The end-to-end build yields the agreed columns with no missing labels."""
    frame = frame_of(
        make_inspection(urn="1"),
        make_inspection(urn="2", inspection_type_grouping="S5 Inspection", overall_effectiveness="3", **{OUTCOME: pd.NA}),
    )
    snapshot = frame_of(make_snapshot_row(urn="1"), make_snapshot_row(urn="2"))

    table = build_label_table("2023/24", frame, snapshot)

    assert table["label_primary"].tolist() == [0, 1]
    assert table["label_primary"].notna().all()
    assert table["years_since_previous_inspection"].gt(9).all()
