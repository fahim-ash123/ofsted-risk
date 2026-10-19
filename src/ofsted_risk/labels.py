"""Building the target population and the decline-risk labels.

The unit of analysis is one inspection of a school whose standing grade
before that inspection was Good. Two labels are produced. The primary label
treats a below-Good graded outcome or an ungraded "Concerns" outcome as at
risk, because both lead Ofsted to prioritise a full inspection. The strict
label counts only confirmed below-Good graded outcomes and is used as a
sensitivity check on that choice.
"""

import pandas as pd

from ofsted_risk import config
from ofsted_risk.ingest import parse_grade, parse_ofsted_date

RELATES = "does_the_previous_inspection_relate_to_the_school_in_its_current_form"

# The published 2022/23 file holds runs of "#" in place of some previous
# inspection numbers, the mark a spreadsheet leaves when a value does not
# fit its column. The values are unrecoverable from that file alone.
MASKED_VALUE = r"#+"
OUTCOME = "outcomes_for_ungraded_and_monitoring_inspections"

LABEL_TABLE_COLUMNS = (
    "year",
    "urn",
    "inspection_number",
    "inspection_start_date",
    "inspection_type_grouping",
    "is_graded",
    "label_primary",
    "label_strict",
    "ofsted_phase",
    "region",
    "local_authority",
    "multi_academy_trust_uid",
    "idaci_quintile",
    "total_number_of_pupils",
    "previous_inspection_number",
    "previous_inspection_start_date",
    "previous_publication_date",
    "relates_to_current_form",
    "urn_at_time_of_previous_inspection",
    "years_since_previous_inspection",
    "prior_ungraded_outcome",
    "ungraded_since_last_graded",
)


def eligible(frame: pd.DataFrame, include_predecessor: bool = False) -> pd.DataFrame:
    """Select inspections of schools that were Good going into the inspection.

    Args:
        frame: One academic year of inspections from ``load_between``.
        include_predecessor: If True, keep inspections whose previous
            graded inspection was of a predecessor school, for example
            before academy conversion. The primary analysis leaves these
            out because the previous report describes a different school.

    Returns:
        The eligible rows, in their original order.
    """
    previous_grade = parse_grade(frame["previous_overall_effectiveness"])
    relates = {"Yes", "No"} if include_predecessor else {"Yes"}
    mask = (
        previous_grade.eq(config.STANDING_GRADE_GOOD).fillna(False)
        & frame[RELATES].isin(relates)
        & frame["inspection_type_grouping"].isin(config.INCLUDED_INSPECTION_TYPES)
        & ~frame["ofsted_phase"].isin(config.EXCLUDED_PHASES)
    )
    return frame.loc[mask].copy()


def assign_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the primary and strict decline-risk labels.

    Args:
        frame: Eligible inspections from ``eligible``.

    Returns:
        A copy with ``is_graded``, ``label_primary`` and ``label_strict``.
        ``label_strict`` is missing for ungraded inspections, because they
        cannot confirm a decline.

    Raises:
        ValueError: If any eligible inspection cannot be labelled, which
            would mean an outcome value the rules have not seen before.
    """
    labelled = frame.copy()
    grade = parse_grade(labelled["overall_effectiveness"])
    outcome = labelled[OUTCOME]
    labelled["is_graded"] = grade.notna()

    primary = pd.Series(pd.NA, index=labelled.index, dtype="Int64")
    primary[grade.isin([3, 4]).fillna(False)] = 1
    primary[grade.isin([1, 2]).fillna(False)] = 0
    primary[outcome.eq(config.CONCERNS_OUTCOME).fillna(False)] = 1
    primary[outcome.isin(config.NO_DECLINE_OUTCOMES)] = 0
    if primary.isna().any():
        unseen = labelled.loc[primary.isna(), [OUTCOME, "overall_effectiveness"]].drop_duplicates()
        raise ValueError(f"Unlabelled inspections with values:\n{unseen.to_string()}")

    labelled["label_primary"] = primary
    labelled["label_strict"] = primary.where(labelled["is_graded"])
    return labelled


def first_per_school(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep each school's first eligible inspection in the year.

    A Concerns visit can be followed by a graded inspection in the same
    year, both pointing back to the same previous report. The first visit
    is the point at which a risk score would have been used.

    Args:
        frame: Labelled inspections for one academic year.

    Returns:
        One row per school. The number of later inspections removed is
        stored in ``attrs["repeat_inspections_dropped"]``.
    """
    ordered = frame.assign(_start=parse_ofsted_date(frame["inspection_start_date"]))
    ordered = ordered.sort_values(["urn", "_start"], kind="stable")
    kept = ordered.drop_duplicates("urn", keep="first").drop(columns="_start")
    kept.attrs["repeat_inspections_dropped"] = int(len(frame) - len(kept))
    return kept.sort_index()


def repair_masked_previous_numbers(frame: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Recover previous inspection numbers masked as "#" in the source file.

    The prior snapshot lists each school's latest graded inspection with
    its number and start date. Where the masked row's previous start date
    equals that date, the inspections are the same one and its number can
    be restored. Anything that cannot be matched stays missing rather than
    being guessed.

    Args:
        frame: Inspections for one academic year.
        snapshot: The "as at" file for 31 August before that year.

    Returns:
        A copy with masked numbers restored where possible. The counts of
        masked, repaired and unresolved values are stored in
        ``attrs["masked_previous_numbers"]``.
    """
    repaired = frame.copy()
    masked = (
        repaired["previous_inspection_number"].astype("string").str.fullmatch(MASKED_VALUE).fillna(False).to_numpy(bool)
    )
    lookup = snapshot.set_index("urn")[["inspection_number_of_latest_graded_inspection", "inspection_start_date"]]
    candidates = lookup.reindex(repaired.loc[masked, "urn"])
    candidates.index = repaired.index[masked]
    same_inspection = (
        candidates["inspection_start_date"]
        .eq(repaired.loc[masked, "previous_inspection_start_date"])
        .fillna(False)
        .astype(bool)
    )
    restored = candidates["inspection_number_of_latest_graded_inspection"].where(same_inspection)
    repaired.loc[masked, "previous_inspection_number"] = restored
    repaired.attrs["masked_previous_numbers"] = {
        "masked": int(masked.sum()),
        "repaired": int(restored.notna().sum()),
        "unresolved": int(restored.isna().sum()),
    }
    return repaired


def prior_ungraded_context(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Extract each school's ungraded history from the snapshot before a year.

    Args:
        snapshot: The "as at 31 August" file for the year before the
            target year, from ``load_as_at``.

    Returns:
        One row per URN with ``latest_graded_number``,
        ``prior_ungraded_outcome`` and ``ungraded_since_last_graded`` as an
        integer count. The outcome is "none" when there was no ungraded
        inspection, and "not_recorded" when one took place but the file
        gives no outcome, which the 2021 snapshot does for most ungraded
        inspections carried out under the earlier framework.
    """
    count = pd.to_numeric(
        snapshot["number_of_ungraded_inspections_since_the_last_graded_inspection"], errors="coerce"
    ).fillna(0)
    recorded = snapshot["ungraded_inspection_overall_outcome"]
    outcome = recorded.where(recorded.notna(), "not_recorded").where(count > 0, "none")
    return pd.DataFrame(
        {
            "urn": snapshot["urn"],
            "latest_graded_number": snapshot["inspection_number_of_latest_graded_inspection"],
            "prior_ungraded_outcome": outcome,
            "ungraded_since_last_graded": count.astype("Int64"),
        }
    )


def attach_prior_ungraded(frame: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    """Join ungraded history known before the target year.

    The snapshot is only trusted where its latest graded inspection is the
    same one the target row calls its previous inspection. Otherwise the
    ungraded history could belong to a different inspection cycle, so the
    outcome is set to "unknown" rather than guessed.

    Args:
        frame: Labelled inspections for one academic year.
        context: Output of ``prior_ungraded_context``.

    Returns:
        A copy with ``prior_ungraded_outcome`` and
        ``ungraded_since_last_graded``. The number of rows marked "unknown"
        is stored in ``attrs["prior_context_unknown"]``.
    """
    merged = frame.merge(context, on="urn", how="left", validate="many_to_one")
    consistent = merged["latest_graded_number"].eq(merged["previous_inspection_number"]).fillna(False)
    merged.loc[~consistent, "prior_ungraded_outcome"] = "unknown"
    merged.loc[~consistent, "ungraded_since_last_graded"] = pd.NA
    merged = merged.drop(columns="latest_graded_number")
    merged.index = frame.index
    merged.attrs["prior_context_unknown"] = int((~consistent).sum())
    return merged


def build_label_table(
    year: str,
    inspections: pd.DataFrame,
    snapshot: pd.DataFrame,
    include_predecessor: bool = False,
) -> pd.DataFrame:
    """Build the labelled population for one target year.

    Args:
        year: Academic year such as "2023/24".
        inspections: That year's inspections from ``load_between``.
        snapshot: The "as at" file for 31 August before the year starts.
        include_predecessor: Passed to ``eligible``.

    Returns:
        One row per school with the columns in ``LABEL_TABLE_COLUMNS``.
        ``attrs`` records the counts removed at each step for reporting.
    """
    repaired = repair_masked_previous_numbers(inspections, snapshot)
    population = eligible(repaired, include_predecessor=include_predecessor)
    labelled = first_per_school(assign_labels(population))
    with_context = attach_prior_ungraded(labelled, prior_ungraded_context(snapshot))

    start = parse_ofsted_date(with_context["inspection_start_date"])
    previous_start = parse_ofsted_date(with_context["previous_inspection_start_date"])
    table = with_context.assign(
        year=year,
        relates_to_current_form=with_context[RELATES].eq("Yes").fillna(False),
        idaci_quintile=pd.to_numeric(
            with_context["the_income_deprivation_affecting_children_index_idaci_quintile"], errors="coerce"
        ).astype("Int64"),
        total_number_of_pupils=pd.to_numeric(with_context["total_number_of_pupils"], errors="coerce").astype("Int64"),
        years_since_previous_inspection=((start - previous_start).dt.days / 365.25).round(2),
    )
    result = table[list(LABEL_TABLE_COLUMNS)].reset_index(drop=True)
    result.attrs = {
        "year": year,
        "eligible_inspections": int(len(population)),
        "repeat_inspections_dropped": labelled.attrs["repeat_inspections_dropped"],
        "prior_context_unknown": with_context.attrs["prior_context_unknown"],
        "masked_previous_numbers": repaired.attrs["masked_previous_numbers"],
    }
    return result
