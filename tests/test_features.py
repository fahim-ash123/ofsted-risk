"""Tests for joining labels, lagged context and text into the modelling table."""

import pandas as pd

from ofsted_risk.features import build_model_table, urban_rural


def labels_frame():
    """Build a two-row label table shaped like the real one, read as text.

    Returns:
        A label table for one 2023/24 inspection and one 2022/23 inspection.
    """
    base = {
        "inspection_number": ["10000001", "10000002"],
        "inspection_start_date": ["15/09/2023", "15/09/2022"],
        "inspection_type_grouping": ["S5 Inspection", "Section 8 (did not convert)"],
        "is_graded": ["True", "False"],
        "label_primary": ["1", "0"],
        "label_strict": ["1", ""],
        "ofsted_phase": ["Primary", "Secondary"],
        "region": ["London", "North East"],
        "local_authority": ["Camden", "Durham"],
        "multi_academy_trust_uid": ["TR001", ""],
        "idaci_quintile": ["5", "1"],
        "total_number_of_pupils": ["210", "900"],
        "previous_inspection_number": ["ITS1", "ITS2"],
        "previous_inspection_start_date": ["10/10/2013", "01/02/2017"],
        "previous_publication_date": ["01/11/2013", "01/03/2017"],
        "relates_to_current_form": ["True", "True"],
        "years_since_previous_inspection": ["9.93", "5.62"],
        "prior_ungraded_outcome": ["School remains Good (Concerns) - S5 Next", "none"],
        "ungraded_since_last_graded": ["1", "0"],
    }
    return pd.DataFrame({"year": ["2023/24", "2022/23"], "urn": ["100001", "100002"],
                         "urn_at_time_of_previous_inspection": ["100001", "100002"], **base})


def test_build_model_table_joins_the_year_before_and_keeps_every_row():
    """Context comes from the lagged year, and no join adds or drops rows."""
    census = {
        "2023/24": pd.DataFrame({"urn": ["100001"], "census_headcount": [205], "census_pct_fsm": [30.0], "census_pct_eal": [40.0]}),
        "2022/23": pd.DataFrame({"urn": ["100002"], "census_headcount": [880], "census_pct_fsm": [10.0], "census_pct_eal": [2.0]}),
    }
    absence = pd.DataFrame(
        {
            "urn": ["100001", "100001", "100002"],
            "time_period": ["202223", "202324", "202122"],
            "absence_overall_pct": [6.0, 99.0, 8.0],
            "absence_persistent_pct": [20.0, 99.0, 25.0],
        }
    )
    gias = pd.DataFrame({"urn": ["100001", "100002"], "urbanrural_name": ["(England/Wales) Urban major conurbation", "Smaller rural: Further from a major town or city"]})
    texts = pd.DataFrame({"report_key": ["100001|10/10/2013"], "improvement_text": ["Improve marking."]})

    table = build_model_table(labels_frame(), texts, census, absence, gias)

    assert len(table) == 2
    assert table["absence_overall_pct"].tolist() == [6.0, 8.0]
    assert table["absence_covid_affected"].tolist() == [0, 1]
    assert table["census_pct_fsm"].tolist() == [30.0, 10.0]
    assert table["urban_rural"].tolist() == ["urban", "rural"]
    assert table["prior_concerns"].tolist() == [1, 0]
    assert table["in_multi_academy_trust"].tolist() == [1, 0]
    assert table["improvement_text"].iloc[0] == "Improve marking."
    assert pd.isna(table["improvement_text"].iloc[1])
    assert table["is_graded"].tolist() == [True, False]


def test_urban_rural_reads_both_classification_systems():
    """The 2011 and newer GIAS codes collapse to the same two values."""
    codes = pd.Series(["(England/Wales) Rural town and fringe", "Urban: Nearer to a major town or city", None])

    assert urban_rural(codes).tolist()[:2] == ["rural", "urban"]
    assert pd.isna(urban_rural(codes).iloc[2])
