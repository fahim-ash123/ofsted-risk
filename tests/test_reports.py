"""Tests for report link parsing, report selection and resuming."""

import json
from datetime import date

import pandas as pd

from ofsted_risk.reports import (
    completed_keys,
    parse_report_links,
    report_targets,
    restore_from_shards,
    save_shard,
    select_report,
)

# Mirrors the structure of a live school page fetched on Deepnote in
# October 2026: the date sits outside the link, a screen-reader span repeats
# the title, and a second <time> holds the publication date.
PAGE = """
<h2 class="heading--sub">All reports</h2>
<ol class="timeline">
  <li class="timeline__day">
    <div class="event">
      <p class="timeline__date"><time>12 September 2023</time></p>
      <span class="event__title heading--sub"><a class="publication-link" target="_blank"
        href="https://files.ofsted.gov.uk/v1/file/50232288">
        School inspection <span class="nonvisual">School inspection, PDF - 09 November 2023</span></a>
        <div class="event__metadata"><span class="filetype">PDF</span> &ndash; Published
        <time pubdate>09 November 2023</time></div></span>
    </div>
  </li>
  <li class="timeline__day">
    <div class="event">
      <p class="timeline__date"><time>17 January 2017</time></p>
      <span class="event__title heading--sub"><a class="publication-link" target="_blank"
        href="https://files.ofsted.gov.uk/v1/file/2652541">
        Short inspection <span class="nonvisual">Short inspection, PDF - 23 February 2017</span></a></span>
    </div>
  </li>
</ol>
<a href="https://reports.ofsted.gov.uk/cookies">Set cookie preferences</a>
"""


# Mirrors a page updated for report cards, fetched on Deepnote in October
# 2026: the date sits inside the link, and the list appears inside an
# HTML-escaped template as well as in the page itself.
NEW_LINK = (
    '<a class="publication-link" target="_blank" href="https://files.ofsted.gov.uk/v1/file/{file_id}">'
    '<time>{listed}</time>: {kind}: (<span class="filetype">PDF</span>) '
    '<span class="nonvisual">{kind}, pdf - 20 April 2026</span></a><div>Published on <time>20 April 2026</time></div>'
)
NEW_ENTRIES = NEW_LINK.format(file_id="50299895", listed="27 January 2026", kind="School inspection") + NEW_LINK.format(
    file_id="2617121", listed="02 November 2016", kind="Full inspection"
)
NEW_PAGE = (
    '<div data-template="' + NEW_ENTRIES.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    + '"></div><ol class="timeline"><li class="timeline__day">' + NEW_ENTRIES + "</li></ol>"
)


def test_parse_report_links_reads_date_kind_and_file():
    """Report links are recognised and navigation links are ignored."""
    links = parse_report_links(PAGE)

    assert [link.file_id for link in links] == ["50232288", "2652541"]
    assert links[0].inspection_date == date(2023, 9, 12)
    assert [link.kind for link in links] == ["School inspection", "Short inspection"]


def test_select_report_prefers_an_exact_date():
    """The report whose listed date equals the start date is chosen."""
    chosen, match = select_report(parse_report_links(PAGE), date(2017, 1, 17))

    assert (chosen.file_id, match) == ("2652541", "exact")


def test_select_report_allows_a_small_gap_but_not_a_large_one():
    """A two-day gap still matches; a ten-day gap does not."""
    links = parse_report_links(PAGE)

    near, near_match = select_report(links, date(2023, 9, 14))
    far, far_match = select_report(links, date(2023, 9, 22))

    assert (near.file_id, near_match) == ("50232288", "within_tolerance")
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


def test_publication_date_is_not_taken_as_the_inspection_date():
    """Only the timeline date counts, never the later publication date."""
    links = parse_report_links(PAGE)

    assert date(2023, 11, 9) not in [link.inspection_date for link in links]


def test_shards_save_only_new_records_and_restore_after_a_restart(tmp_path):
    """Batches become separate shards, and a lost local file is rebuilt.

    Args:
        tmp_path: Temporary folder provided by pytest.
    """
    local = tmp_path / "local" / "reports.jsonl"
    shards = tmp_path / "project" / "reports"
    local.parent.mkdir()
    local.write_text('{"report_key": "1", "status": "ok"}\n{"report_key": "2", "status": "ok"}\n', encoding="utf-8")

    first = save_shard(local, 0, shards)
    with local.open("a", encoding="utf-8") as handle:
        handle.write('{"report_key": "3", "status": "ok"}\n')
    second = save_shard(local, 2, shards)
    nothing_new = save_shard(local, 3, shards)
    local.unlink()
    restored = restore_from_shards(shards, local)

    assert first.read_text(encoding="utf-8").count("report_key") == 2
    assert second.read_text(encoding="utf-8").count("report_key") == 1
    assert nothing_new is None
    assert restored == 3
    assert completed_keys(local) == {"1", "2", "3"}


def test_parse_report_links_reads_the_report_card_layout_once():
    """Dates inside the link are read, and the escaped repeat is not double counted."""
    links = parse_report_links(NEW_PAGE)

    assert [link.file_id for link in links] == ["50299895", "2617121"]
    assert [link.inspection_date for link in links] == [date(2026, 1, 27), date(2016, 11, 2)]
    assert [link.kind for link in links] == ["School inspection", "Full inspection"]


def test_parse_report_links_reads_a_list_that_only_exists_escaped():
    """A list held only inside the escaped template is still found."""
    escaped_only = NEW_PAGE.split('<ol class="timeline">')[0]

    assert [link.file_id for link in parse_report_links(escaped_only)] == ["50299895", "2617121"]


def test_completed_keys_retries_empty_pages_and_uses_the_latest_record(tmp_path):
    """An empty page is retried, and a later success settles the key.

    Args:
        tmp_path: Temporary folder provided by pytest.
    """
    output = tmp_path / "reports.jsonl"
    lines = [
        {"report_key": "1|a", "status": "empty_page"},
        {"report_key": "2|b", "status": "empty_page"},
        {"report_key": "2|b", "status": "ok"},
        {"report_key": "3|c", "status": "no_match"},
    ]
    output.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    assert completed_keys(output) == {"2|b", "3|c"}
