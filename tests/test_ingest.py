"""Tests for loading, header normalisation and code handling."""

import pandas as pd
import pytest

from ofsted_risk import config
from ofsted_risk.ingest import non_grade_codes, normalise_header, parse_grade, read_ofsted_csv


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Web Link (opens in new window)", "web_link"),
        ("Web Link", "web_link"),
        ("Academy Sponsor Name", "academy_sponsor_name"),
        ("Safeguarding is effective?", "safeguarding_is_effective"),
        ("Number of warning notices issued in 2024/25 academic year", "number_of_warning_notices_issued_in_academic_year"),
        ("% of pupils known to be eligible for free school meals", "pct_of_pupils_known_to_be_eligible_for_free_school_meals"),
        ("Section 8 inspection of a good and outstanding school overall outcome", "ungraded_inspection_overall_outcome"),
        ("headcount total girls", "headcount_total_female"),
    ],
)
def test_normalise_header_maps_year_variants_to_one_name(raw, expected):
    """Header wording that changed between releases ends up identical.

    Args:
        raw: Header as published.
        expected: Canonical name the pipeline relies on.
    """
    assert normalise_header(raw) == expected


def test_read_ofsted_csv_drops_blank_rows_and_missing_markers(tmp_path):
    """Trailing blank rows go, NULL becomes missing and grade codes stay text.

    The 2022/23 file carries thousands of empty rows and every file uses the
    literal string NULL, so both must be handled before counting anything.

    Args:
        tmp_path: Temporary folder provided by pytest.
    """
    path = tmp_path / "inspections.csv"
    text = "URN,School name,Overall effectiveness,Previous inspection number\n"
    text += "100001,St Mary\u2019s School,9,NULL\n,,,\n,,,\n"
    path.write_bytes(text.encode(config.OFSTED_ENCODING))

    frame = read_ofsted_csv(path)

    assert len(frame) == 1
    assert frame.attrs["blank_rows_dropped"] == 2
    assert frame.loc[0, "school_name"] == "St Mary\u2019s School"
    assert frame.loc[0, "overall_effectiveness"] == "9"
    assert pd.isna(frame.loc[0, "previous_inspection_number"])
    assert all(isinstance(dtype, pd.StringDtype) for dtype in frame.dtypes)


def test_read_ofsted_csv_rejects_files_without_urn(tmp_path):
    """A file with no URN column cannot be joined, so loading fails loudly.

    Args:
        tmp_path: Temporary folder provided by pytest.
    """
    path = tmp_path / "wrong.csv"
    path.write_text("School name\nA school\n", encoding=config.OFSTED_ENCODING)

    with pytest.raises(KeyError):
        read_ofsted_csv(path)


def test_parse_grade_treats_codes_as_missing():
    """Only 1 to 4 are grades; 9, 0, Not judged and missing are not."""
    series = pd.Series(["1", "4", "9", "0", "Not judged", pd.NA])

    result = parse_grade(series)

    assert result.tolist()[:2] == [1, 4]
    assert result.iloc[2:].isna().all()


def test_non_grade_codes_reports_each_code():
    """The report needs every non-grade code counted, including the stray 0."""
    series = pd.Series(["2", "9", "9", "0", pd.NA])

    assert non_grade_codes(series) == {"9": 2, "0": 1, "<missing>": 1}


def test_year_mappings_cover_every_modelled_year():
    """Each target and drift year has a census file and an absence period."""
    years = (*config.TARGET_YEARS, config.DRIFT_YEAR)

    for year in years:
        assert year in config.CENSUS_FILES
        assert year in config.ABSENCE_PERIOD
        assert year in config.STANDING_SNAPSHOT


def test_absence_period_is_the_year_before_the_target():
    """Absence must come from the year before, or the feature would leak."""
    for year, period in config.ABSENCE_PERIOD.items():
        start = int(year[:4])
        assert period == f"{start - 1}{str(start)[2:]}"
