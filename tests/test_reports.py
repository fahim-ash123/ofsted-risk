"""Tests for report link parsing, report selection and resuming."""

import json
from datetime import date

import pandas as pd

from ofsted_risk.reports import completed_keys, parse_report_links, report_targets, select_report

# Mirrors the link text on a live school page checked in October 2026.
PAGE = """
<ol>
<li><a href="https://files.ofsted.gov.uk/v1/file/50189058">17 May 2022: School inspection: (PDF)
School inspection, PDF - 21 July 2022</a> Published on 21 July 2022</li>
<li><a href="https://files.ofsted.gov.uk/v1/file/2652541">17 January 2017: Short inspection: (PDF)
Short inspection, PDF - 23 February 2017</a></li>
<li><a href="https://reports.ofsted.gov.uk/cookies">Set cookie preferences</a></li>
</ol>
"""


def test_parse_report_links_reads_date_kind_and_file():
    """Report links are recognised and navigation links are ignored."""
    links = parse_report_links(PAGE)

    assert [link.file_id for link in links] == ["50189058", "2652541"]
    assert links[0].inspection_date == date(2022, 5, 17)
    assert links[1].kind == "Short inspection"


def test_select_report_prefers_an_exact_date():
    """The report whose listed date equals the start date is chosen."""
    chosen, match = select_report(parse_report_links(PAGE), date(2017, 1, 17))

    assert (chosen.file_id, match) == ("2652541", "exact")


def test_select_report_allows_a_small_gap_but_not_a_large_one():
    """A two-day gap still matches; a ten-day gap does not."""
    links = parse_report_links(PAGE)

    near, near_match = select_report(links, date(2022, 5, 19))
    far, far_match = select_report(links, date(2022, 5, 27))

    assert (near.file_id, near_match) == ("50189058", "within_tolerance")
    assert (far, far_match) == (None, "no_match")


def test_report_targets_deduplicates_and_puts_test_year_first():
    """Each previous report is listed once, with the test year leading."""
    table = pd.DataFrame(
        {
            "year": ["2021/22", "2023/24", "2023/24"],
            "urn_at_time_of_previous_inspection": ["1", "2", "1"],
            "previous_inspection_start_date": ["10/10/2013", "01/02/2018", "10/10/2013"],
        }
    )

    targets = report_targets([table], first_year="2023/24")

    assert targets["report_key"].tolist() == ["2|01/02/2018", "1|10/10/2013"]


def test_completed_keys_reads_every_recorded_status(tmp_path):
    """Failed lookups count as done so a restart does not repeat them.

    Args:
        tmp_path: Temporary folder provided by pytest.
    """
    output = tmp_path / "reports.jsonl"
    lines = [{"report_key": "1|a", "status": "ok"}, {"report_key": "2|b", "status": "no_match"}]
    output.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    assert completed_keys(output) == {"1|a", "2|b"}
    assert completed_keys(tmp_path / "missing.jsonl") == set()
