"""Tests for finding and cleaning the improvement section.

The fixtures are trimmed from real reports downloaded in October 2026, one
per framework layout, with names outside the section left out.
"""

import pandas as pd

from ofsted_risk.text import build_text_table, clean_section, find_improvement_section, report_year

REPORT_2011 = """What does the school need to do to improve further?
 Strengthen teaching so that the level of challenge and progress in lessons is consistently good or better.
 Extend the variety of opportunities across the curriculum for pupils to write for a range of purposes.
Outcomes for individuals and groups of pupils 1
All groups of pupils enjoy learning and their achievement is good.

Inspection judgements
Overall effectiveness: how good is the school? 2"""

REPORT_2012 = """What does the school need to do to improve further?
Inspection grades: 1 is outstanding, 2 is good, 3 is satisfactory,
and 4 is inadequate
Please turn to the glossary for a description of the grades and inspection terms
 By June 2013, raise attainment and accelerate progress in writing at Key Stage 2.
Main report
Achievement of pupils
Children enter the Early Years Foundation Stage at the broadly expected starting points."""

REPORT_ACADEMY = """This is a good academy
What does the academy need to do to improve further?
 Make marking consistent in Key Stage 3 and the sixth form.
Inspection judgements
The achievement of pupils is good"""

REPORT_2013 = """This is a good school.
What does the school need to do to improve further?
\uf0b7 Improve the quality of teaching from good to outstanding by ensuring that:
\uf0b7 teachers' marking always includes clear steps on how it can be improved

  Inspection report:  West Twyford Primary School, 5–6 December 2013 4 of 9

Inspection judgements
The achievement of pupils is good"""

REPORT_2017 = """What does the school need to do to improve further?

\uf0b7 Strengthen outcomes further by:
– increasing pupils' attendance.
\uf0b7 Ensure that governors improve their knowledge and oversight of attendance.

Inspection report: Wildground Infant School, 12–13 September 2017

Page 3 of 11

Inspection judgements
Effectiveness of leadership and management Outstanding"""

REPORT_2019 = """What does the school need to do to improve further?
\uf0b7 Strengthen the impact of leadership by ensuring that leaders are consistently effective.

Inspection report: Kingsbury School - A Specialist Science and Mathematics Academy,
                              9–10 April 2019

Page 3 of 10

Inspection judgements
The headteacher has steered the school through challenging times."""

REPORT_2020 = """What does the school need to do to improve?

(Information for the school and appropriate authority)

\uf0b7 In some subjects, such as art and music, the sequence of learning is not yet clear enough.

How can I feed back my views?

You can use Ofsted Parent View to give Ofsted your opinion."""


def test_each_framework_layout_yields_only_the_improvement_points():
    """All four layouts end at the right heading and keep only the section."""
    expected = {
        REPORT_2011: ("outcomes for individuals and groups of pupils", "write for a range of purposes", "enjoy learning"),
        REPORT_2012: ("main report", "accelerate progress in writing", "broadly expected"),
        REPORT_ACADEMY: ("inspection judgements", "Make marking consistent", "achievement of pupils"),
        REPORT_2013: ("inspection judgements", "Improve the quality of teaching", "achievement of pupils"),
        REPORT_2017: ("inspection judgements", "Strengthen outcomes further", "leadership and management"),
        REPORT_2019: ("inspection judgements", "Strengthen the impact of leadership", "steered the school"),
        REPORT_2020: ("how can i feed back my views", "sequence of learning", "Parent View"),
    }
    for report, (marker, kept, dropped) in expected.items():
        section, found_marker, capped = find_improvement_section(report)
        cleaned, _ = clean_section(section)
        assert (found_marker, capped) == (marker, False)
        assert kept in cleaned
        assert dropped not in cleaned


def test_cleaning_removes_school_identity_page_numbers_and_notes():
    """Headers naming the school, page numbers and audience notes are removed."""
    for report in (REPORT_2012, REPORT_2013, REPORT_2017, REPORT_2019, REPORT_2020):
        cleaned, _ = clean_section(find_improvement_section(report)[0])
        for leaked in ("Inspection report", "Twyford", "Wildground", "Kingsbury", "of 11", "appropriate authority", "glossary"):
            assert leaked not in cleaned


def test_titled_names_are_replaced_and_counted():
    """A titled name inside the section becomes a placeholder."""
    cleaned, removed = clean_section("Ensure that Mrs Jane Smith and Mr Patel review attendance.")

    assert cleaned == "Ensure that [name] and [name] review attendance."
    assert removed == 2


def test_missing_heading_and_missing_end_are_reported():
    """A report with no heading gives no text, and an unterminated one is capped."""
    assert find_improvement_section("A report with no such section.") == (None, "no_heading", False)
    section, marker, capped = find_improvement_section("What does the school need to do to improve? " + "x" * 9000)

    assert (marker, capped) == ("length_cap", True)
    assert len(section) == 4000


def test_build_text_table_keeps_unmatched_reports_without_text():
    """Every report gets a row, and only downloaded ones carry text."""
    records = [
        {"report_key": "1|a", "status": "ok", "kind": "Full inspection: Good", "listed_date": "2017-09-12", "pages": [REPORT_2017]},
        {"report_key": "2|b", "status": "no_match", "links_found": 3},
    ]

    table = build_text_table(records)

    assert table["report_key"].tolist() == ["1|a", "2|b"]
    assert table.loc[0, "characters"] > 50
    assert pd.isna(table.loc[1, "improvement_text"])


def test_report_year_handles_missing_dates():
    """Unmatched reports have no date, which must not stop the build."""
    assert report_year("2017-09-12") == "2017"
    assert report_year(float("nan")) is None
    assert report_year(None) is None


def test_headings_split_across_lines_are_still_found():
    """A heading broken over two lines by PDF extraction is still recognised."""
    report = "What does the school need to do to\nimprove further?\n Raise attainment in writing.\nInspection\njudgements\nMore text"

    section, marker, _ = find_improvement_section(report)

    assert marker == "inspection judgements"
    assert clean_section(section)[0] == "Raise attainment in writing."
