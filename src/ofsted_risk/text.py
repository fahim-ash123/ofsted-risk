"""Extracting and cleaning the improvement section of each previous report.

Every framework since 2010 includes a section headed "What does the school
need to do to improve (further)?". It is the inspector's own statement of
where the school was weakest, so it is the text the models use. The section
ends at a framework-specific heading: "Outcomes for individuals and groups
of pupils" in reports from the framework used until mid-2012, "Inspection
judgements" from then until 2019, "How can I feed back my views?" under the
2019 framework, and "School details" or "Information about this inspection"
as later fallbacks. The earliest of these after the heading is used.

Cleaning removes what would let a model recognise a school instead of
learning from what inspectors wrote: page headers carrying the school's
name and inspection dates, page numbers, and any titled personal names.
"""

import re
from typing import Iterable, Optional

import pandas as pd

START_HEADINGS = (
    "what does the school need to do to improve",
    "what the school needs to do to improve",
)
END_MARKERS = (
    "outcomes for individuals and groups of pupils",
    "inspection judgements",
    "how can i feed back my views",
    "school details",
    "information about this inspection",
)

# Used only when no end marker follows the heading, so that one malformed
# report cannot pull pages of unrelated text into the model.
MAX_SECTION_CHARACTERS = 4000

HEADER_LINE = re.compile(r"^\s*Inspection report\s*:", re.IGNORECASE)
PAGE_LINE = re.compile(r"^\s*(Page\s+)?\d+\s+of\s+\d+\s*$", re.IGNORECASE)
DATE_LINE = re.compile(r"^\s*\d{1,2}(\s*[–-]\s*\d{1,2})?\s+[A-Za-z]+\s+\d{4}(\s+\d+(\s+of\s+\d+)?)?\s*$")
AUDIENCE_NOTE = re.compile(r"\(Information for the school and appropriate authority\)", re.IGNORECASE)
TITLED_NAME = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)?")
# PDF bullets arrive as characters from the Unicode private-use area.
BULLET = re.compile(r"[\ue000-\uf8ff•]")


def find_improvement_section(text: str) -> tuple:
    """Locate the improvement section in a report's full text.

    Args:
        text: All pages of one report joined together.

    Returns:
        A triple of the section text (without its heading), the end marker
        that closed it, and whether the length cap was used instead. The
        section is None, with marker "no_heading", if no heading is found.
    """
    lowered = text.lower()
    starts = [lowered.find(heading) for heading in START_HEADINGS if heading in lowered]
    if not starts:
        return None, "no_heading", False
    start = min(starts)
    question = text.find("?", start)
    body_start = question + 1 if 0 <= question - start < 120 else start
    ends = [
        (position, marker)
        for marker in END_MARKERS
        if (position := lowered.find(marker, body_start)) >= 0
    ]
    if ends:
        end, marker = min(ends)
        return text[body_start:end], marker, False
    return text[body_start : body_start + MAX_SECTION_CHARACTERS], "length_cap", True


def clean_section(section: str) -> tuple:
    """Remove headers, page numbers, bullets and titled names from a section.

    Args:
        section: Raw section text from ``find_improvement_section``.

    Returns:
        A pair of the cleaned text on a single line and the number of
        titled names replaced with "[name]".
    """
    kept = [
        line
        for line in section.splitlines()
        if not (HEADER_LINE.match(line) or PAGE_LINE.match(line) or DATE_LINE.match(line))
    ]
    joined = AUDIENCE_NOTE.sub(" ", "\n".join(kept))
    joined, names_removed = TITLED_NAME.subn("[name]", joined)
    joined = BULLET.sub(" ", joined).replace("– ", " ")
    return " ".join(joined.split()), names_removed


def build_text_table(records: Iterable[dict]) -> pd.DataFrame:
    """Build one row of cleaned improvement text per downloaded report.

    Args:
        records: Latest download record per report, as stored in the shards.

    Returns:
        One row per report with ``report_key``, ``status``, ``kind``,
        ``listed_date``, ``improvement_text``, ``end_marker``, ``capped``,
        ``names_removed`` and ``characters``. Reports that were not
        downloaded keep their status and have no text.
    """
    rows = []
    for record in records:
        row = {
            "report_key": record["report_key"],
            "status": record["status"],
            "kind": record.get("kind"),
            "listed_date": record.get("listed_date"),
            "improvement_text": None,
            "end_marker": None,
            "capped": False,
            "names_removed": 0,
        }
        if record["status"] == "ok":
            section, marker, capped = find_improvement_section(" \n".join(record["pages"]))
            row.update(end_marker=marker, capped=capped)
            if section is not None:
                row["improvement_text"], row["names_removed"] = clean_section(section)
        rows.append(row)
    table = pd.DataFrame(rows)
    table["characters"] = table["improvement_text"].str.len().fillna(0).astype(int)
    return table


def report_year(listed_date: Optional[str]) -> Optional[str]:
    """Return the calendar year of a report's listed inspection date.

    Args:
        listed_date: ISO date string, or a missing value for reports that
            were not downloaded.

    Returns:
        The four-digit year, or None.
    """
    return listed_date[:4] if isinstance(listed_date, str) and listed_date else None
