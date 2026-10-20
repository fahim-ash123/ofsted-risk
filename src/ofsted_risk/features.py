"""Joining labels, school context and report text into one modelling table.

Every feature must have been knowable before the target inspection began.
Census and absence therefore come from the year before the target year, the
prior ungraded history from the snapshot taken before the year started, and
the text from the previous graded report. Fields describing the target
inspection itself, such as whether it was graded, are kept for analysis but
never used as features, because the choice of inspection type is Ofsted's
own judgement of risk.
"""

from typing import Dict

import pandas as pd

from ofsted_risk import config

# Columns kept for joining, slicing and evaluation, never fed to a model.
ID_COLUMNS = ("year", "urn", "inspection_number", "report_key", "local_authority")
TARGET_COLUMNS = ("label_primary", "label_strict", "is_graded")

CATEGORICAL_FEATURES = ("ofsted_phase", "region", "urban_rural")
NUMERIC_FEATURES = (
    "idaci_quintile",
    "total_number_of_pupils",
    "in_multi_academy_trust",
    "years_since_previous_inspection",
    "ungraded_since_last_graded",
    "prior_concerns",
    "census_pct_fsm",
    "census_pct_eal",
    "absence_overall_pct",
    "absence_persistent_pct",
    "text_characters",
)
# Kept in the table for analysis, not used as a feature: both training years
# draw on COVID-affected absence (2020/21 and 2021/22) and the test year does
# not, so the flag is constant within each split and carries no signal.
ANALYSIS_COLUMNS = ("absence_covid_affected",)
TEXT_FEATURE = "improvement_text"


def urban_rural(classification: pd.Series) -> pd.Series:
    """Collapse GIAS rural-urban codes into "urban" or "rural".

    GIAS mixes two classification systems, the 2011 codes such as
    "(England/Wales) Urban city and town" and newer codes such as "Smaller
    rural: Nearer to a major town or city". Only the urban or rural split
    is common to both.

    Args:
        classification: The GIAS ``UrbanRural (name)`` column.

    Returns:
        "urban", "rural", or missing where the code is absent or unclear.
    """
    lowered = classification.astype("string").str.lower()
    result = pd.Series(pd.NA, index=classification.index, dtype="string")
    result[lowered.str.contains("rural", na=False)] = "rural"
    result[lowered.str.contains("urban", na=False)] = "urban"
    return result


def build_model_table(
    labels: pd.DataFrame,
    texts: pd.DataFrame,
    census: Dict[str, pd.DataFrame],
    absence: pd.DataFrame,
    gias: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per labelled inspection with every feature attached.

    Args:
        labels: Label table from ``build_label_table``, read as text.
        texts: Text table from ``build_text_table``.
        census: Census context by target year, from ``load_census``.
        absence: School-level absence for all years, from ``load_absence``.
        gias: Establishment register from ``load_gias``.

    Returns:
        The label rows in their original order with identifier, target,
        feature and text columns. The row count always equals the label
        table's, because every join is a left join on a unique key.
    """
    table = labels.copy()
    table["report_key"] = table["urn_at_time_of_previous_inspection"] + "|" + table["previous_inspection_start_date"]
    for column in ("label_primary", "label_strict", "idaci_quintile", "total_number_of_pupils"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    for column in ("years_since_previous_inspection", "ungraded_since_last_graded"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["is_graded"] = table["is_graded"].astype(str).str.lower().eq("true")
    table["in_multi_academy_trust"] = table["multi_academy_trust_uid"].astype(str).str.strip().ne("").astype(int)
    table["prior_concerns"] = table["prior_ungraded_outcome"].astype(str).str.contains("Concerns").astype(int)

    census_rows = pd.concat(
        [frame.assign(year=year) for year, frame in census.items() if year in set(table["year"])], ignore_index=True
    )
    table = table.merge(census_rows, on=["year", "urn"], how="left", validate="many_to_one")

    periods = pd.DataFrame({"year": list(config.ABSENCE_PERIOD), "time_period": list(config.ABSENCE_PERIOD.values())})
    absence_rows = absence.merge(periods, on="time_period")[
        ["year", "urn", "time_period", "absence_overall_pct", "absence_persistent_pct"]
    ]
    table = table.merge(absence_rows, on=["year", "urn"], how="left", validate="many_to_one")
    table["absence_covid_affected"] = table["time_period"].isin(config.COVID_AFFECTED_PERIODS).astype(int)

    place = gias[["urn", "urbanrural_name"]].assign(urban_rural=lambda frame: urban_rural(frame["urbanrural_name"]))
    table = table.merge(place[["urn", "urban_rural"]], on="urn", how="left", validate="many_to_one")

    text_rows = texts[["report_key", TEXT_FEATURE]]
    table = table.merge(text_rows, on="report_key", how="left", validate="many_to_one")
    table["text_characters"] = table[TEXT_FEATURE].str.len().fillna(0).astype(int)

    columns = [*ID_COLUMNS, *TARGET_COLUMNS, *CATEGORICAL_FEATURES, *NUMERIC_FEATURES, *ANALYSIS_COLUMNS, TEXT_FEATURE]
    if len(table) != len(labels):
        raise ValueError(f"Join changed the row count from {len(labels)} to {len(table)}")
    return table[columns]


def missing_by_year(table: pd.DataFrame) -> pd.DataFrame:
    """Share of missing values in each feature, by target year.

    Args:
        table: Output of ``build_model_table``.

    Returns:
        Years as rows, features as columns, values as proportions.
    """
    features = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, TEXT_FEATURE]
    return table.groupby("year")[features].apply(lambda frame: frame.isna().mean()).round(4)
