"""Row builders for small synthetic Ofsted frames used across tests.

The real files are too large and change between releases, so the tests
build frames that reproduce only the behaviours being checked. Every row
defaults to an eligible ungraded inspection of a previously Good primary
school, and each test overrides only what it is about.
"""

import pandas as pd

from ofsted_risk.labels import OUTCOME, RELATES


def make_inspection(**overrides) -> dict:
    """Build one inspection row with eligible defaults.

    Args:
        **overrides: Column values to replace the defaults.

    Returns:
        A mapping from column name to value for one inspection.
    """
    row = {
        "urn": "100001",
        "inspection_number": "10000001",
        "inspection_start_date": "15/09/2023",
        "inspection_type_grouping": "Section 8 (did not convert)",
        "overall_effectiveness": "9",
        OUTCOME: "School remains Good",
        "previous_overall_effectiveness": "2",
        RELATES: "Yes",
        "ofsted_phase": "Primary",
        "region": "London",
        "local_authority": "Camden",
        "multi_academy_trust_uid": pd.NA,
        "the_income_deprivation_affecting_children_index_idaci_quintile": "3",
        "total_number_of_pupils": "200",
        "previous_inspection_number": "ITS000001",
        "previous_inspection_start_date": "10/10/2013",
        "previous_publication_date": "01/11/2013",
        "urn_at_time_of_previous_inspection": "100001",
    }
    row.update(overrides)
    return row


def make_snapshot_row(**overrides) -> dict:
    """Build one "as at" snapshot row matching the default inspection.

    Args:
        **overrides: Column values to replace the defaults.

    Returns:
        A mapping from column name to value for one school.
    """
    row = {
        "urn": "100001",
        "inspection_number_of_latest_graded_inspection": "ITS000001",
        "inspection_start_date": "10/10/2013",
        "overall_effectiveness": "2",
        "ungraded_inspection_overall_outcome": pd.NA,
        "number_of_ungraded_inspections_since_the_last_graded_inspection": "0",
    }
    row.update(overrides)
    return row
