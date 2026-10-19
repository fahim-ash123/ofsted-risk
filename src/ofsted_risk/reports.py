"""Downloading the previous inspection report for each labelled school.

Each school's page on reports.ofsted.gov.uk lists its reports as a
timeline. Every entry holds the inspection date in a ``timeline__date``
paragraph, followed by a link to the PDF on files.ofsted.gov.uk whose
visible text names the report type, such as "School inspection". The
previous inspection start date in the label table is matched against the
timeline date to pick the right PDF.

Only the extracted page text is kept, one JSON line per report. The
project folder on Deepnote is a slow network drive, so lines are written
to the machine's local disk and copied to the project folder in batches as
small shard files. The run can stop and resume at any point, and cleaning
can be redone later without downloading again. Requests are spaced out
because the site's terms ask users not to restrict others' use of it.
"""

import io
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd

PROVIDER_URL = "https://reports.ofsted.gov.uk/inspection-reports/find-inspection-report/provider/ELS/{urn}"
FILE_LINK = re.compile(r"files\.ofsted\.gov\.uk/v1/file/(\d+)")
USER_AGENT = "NU London apprenticeship coursework (Ofsted risk study); low-rate research use"

# Listed dates occasionally differ from the inspection data by a day or
# two, for example where an inspection spans a weekend. Wider gaps are
# treated as no match rather than risk taking the wrong report.
DATE_TOLERANCE_DAYS = 3


@dataclass
class ReportLink:
    """One report listed on a school's page.

    Attributes:
        file_id: Identifier in the files.ofsted.gov.uk address.
        inspection_date: Date the timeline gives for the inspection.
        kind: Report type as listed, such as "School inspection".
    """

    file_id: str
    inspection_date: date
    kind: str


def _classes(attrs: list) -> List[str]:
    """Return the CSS classes of an element.

    Args:
        attrs: Element attributes as name and value pairs.

    Returns:
        The class names, empty if the element has none.
    """
    return (dict(attrs).get("class") or "").split()


class _TimelineCollector(HTMLParser):
    """Walk the report timeline, pairing each PDF link with its date.

    The date sits in a ``<time>`` element inside ``p.timeline__date``.
    Another ``<time>`` gives the publication date further on, so only the
    one inside that paragraph is used. Text inside ``span.nonvisual`` is a
    screen-reader duplicate and is left out of the report type.
    """

    def __init__(self) -> None:
        """Start with no date and no links collected."""
        super().__init__()
        self.links: List[tuple] = []
        self._in_date_paragraph = False
        self._in_date_time = False
        self._date_text: List[str] = []
        self._current_date: Optional[str] = None
        self._href: Optional[str] = None
        self._kind: List[str] = []
        self._hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        """Track entry into the date paragraph, report links and hidden text.

        Args:
            tag: Element name.
            attrs: Element attributes as name and value pairs.
        """
        if tag == "p" and "timeline__date" in _classes(attrs):
            self._in_date_paragraph = True
        elif tag == "time" and self._in_date_paragraph:
            self._in_date_time = True
            self._date_text = []
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._kind = []
            self._hidden_depth = 0
        elif tag == "span" and self._href is not None and (self._hidden_depth or "nonvisual" in _classes(attrs)):
            self._hidden_depth += 1

    def handle_data(self, data: str) -> None:
        """Keep date text and the visible text of the current link.

        Args:
            data: Text content.
        """
        if self._in_date_time:
            self._date_text.append(data)
        elif self._href is not None and not self._hidden_depth:
            self._kind.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Close the date, hidden text or link, storing finished links.

        Args:
            tag: Element name.
        """
        if tag == "time" and self._in_date_time:
            self._in_date_time = False
            self._current_date = " ".join("".join(self._date_text).split())
        elif tag == "p" and self._in_date_paragraph:
            self._in_date_paragraph = False
        elif tag == "span" and self._hidden_depth:
            self._hidden_depth -= 1
        elif tag == "a" and self._href is not None:
            kind = " ".join("".join(self._kind).split())
            self.links.append((self._href, self._current_date, kind))
            self._href = None


def parse_report_links(html: str) -> List[ReportLink]:
    """Extract every dated report link from a school's page.

    Args:
        html: Page source from reports.ofsted.gov.uk.

    Returns:
        One entry per PDF link that follows a timeline date. Other links,
        such as navigation, are ignored.
    """
    collector = _TimelineCollector()
    collector.feed(html)
    links = []
    for href, listed, kind in collector.links:
        file_match = FILE_LINK.search(href or "")
        if not file_match or not listed:
            continue
        try:
            inspection_date = datetime.strptime(listed, "%d %B %Y").date()
        except ValueError:
            continue
        links.append(ReportLink(file_match.group(1), inspection_date, kind))
    return links


def select_report(links: Iterable[ReportLink], start_date: date) -> tuple:
    """Pick the report for a given inspection start date.

    Args:
        links: Reports listed on the school's page.
        start_date: Previous inspection start date from the label table.

    Returns:
        A pair of the chosen link, or None, and how it was matched:
        "exact", "within_tolerance" or "no_match".
    """
    best, best_gap = None, None
    for link in links:
        gap = abs((link.inspection_date - start_date).days)
        if gap == 0:
            return link, "exact"
        if gap <= DATE_TOLERANCE_DAYS and (best_gap is None or gap < best_gap):
            best, best_gap = link, gap
    return (best, "within_tolerance") if best else (None, "no_match")


def report_targets(label_tables: Iterable[pd.DataFrame], first_year: str) -> pd.DataFrame:
    """List the distinct previous reports needed, test year first.

    A school can appear in more than one year with the same previous
    report, so each report is fetched once. The test year comes first so
    that a partial run still covers the data used for evaluation.

    Args:
        label_tables: Label tables from ``build_label_table``.
        first_year: Year to download before the others.

    Returns:
        One row per report with ``report_key``,
        ``urn_at_time_of_previous_inspection`` and
        ``previous_inspection_start_date``.
    """
    combined = pd.concat(label_tables, ignore_index=True)
    combined["_order"] = combined["year"].ne(first_year).astype(int)
    combined = combined.sort_values(["_order", "year"], kind="stable")
    targets = combined.drop_duplicates(["urn_at_time_of_previous_inspection", "previous_inspection_start_date"])
    targets = targets[["urn_at_time_of_previous_inspection", "previous_inspection_start_date"]].copy()
    targets["report_key"] = (
        targets["urn_at_time_of_previous_inspection"] + "|" + targets["previous_inspection_start_date"]
    )
    return targets.reset_index(drop=True)


def completed_keys(output: Path) -> Set[str]:
    """Read which reports an earlier run has already recorded.

    Args:
        output: JSON Lines file written by ``download_reports``.

    Returns:
        The report keys present, whatever their status, so failed lookups
        are not retried on every restart. Delete their lines to retry them.
    """
    if not output.exists():
        return set()
    with output.open(encoding="utf-8") as handle:
        return {json.loads(line)["report_key"] for line in handle if line.strip()}


def restore_from_shards(shard_dir: Path, local: Path) -> int:
    """Rebuild the local results file from shards saved in the project folder.

    After a machine restart the local disk is empty, so earlier progress is
    copied back before the run resumes. An existing local file is kept if
    it already holds at least as many lines as the shards.

    Args:
        shard_dir: Project folder holding ``part_*.jsonl`` shard files.
        local: Results file on the machine's local disk.

    Returns:
        The number of records available locally after restoring.
    """
    shards = sorted(shard_dir.glob("part_*.jsonl")) if shard_dir.exists() else []
    saved = [line for shard in shards for line in shard.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing = local.read_text(encoding="utf-8").splitlines() if local.exists() else []
    if len(saved) > len(existing):
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text("".join(f"{line}\n" for line in saved), encoding="utf-8")
        return len(saved)
    return len(existing)


def save_shard(local: Path, start_line: int, shard_dir: Path) -> Optional[Path]:
    """Copy records added since ``start_line`` into a new shard file.

    Writing one new file per batch keeps the number of slow operations on
    the project drive small, and never rewrites what is already saved.

    Args:
        local: Results file on the machine's local disk.
        start_line: Number of lines already saved in earlier shards.
        shard_dir: Project folder to write the shard into.

    Returns:
        The shard written, or None if there were no new records.
    """
    lines = local.read_text(encoding="utf-8").splitlines()[start_line:] if local.exists() else []
    if not lines:
        return None
    shard_dir.mkdir(parents=True, exist_ok=True)
    number = len(list(shard_dir.glob("part_*.jsonl"))) + 1
    shard = shard_dir / f"part_{number:04d}.jsonl"
    temporary = shard.with_suffix(".tmp")
    temporary.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    shutil.move(str(temporary), str(shard))
    return shard


def extract_pages(pdf_bytes: bytes) -> List[str]:
    """Extract the text of each page of a report PDF.

    Args:
        pdf_bytes: Downloaded PDF content.

    Returns:
        One string per page, in order.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return [page.extract_text() or "" for page in reader.pages]


def download_reports(
    targets: pd.DataFrame,
    output: Path,
    delay: float = 1.0,
    limit: Optional[int] = None,
    progress_every: int = 50,
) -> dict:
    """Fetch and store the text of each target report, resuming safely.

    Every outcome is written as one JSON line straight away, including
    failures, so stopping the run loses at most the report in progress.

    Args:
        targets: Output of ``report_targets``.
        output: JSON Lines file to append to.
        delay: Seconds to wait before each request.
        limit: Stop after this many new reports, for a trial run.
        progress_every: Print a progress line after this many reports.

    Returns:
        Counts of each status recorded during this run.
    """
    import requests

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    done = completed_keys(output)
    pending = targets[~targets["report_key"].isin(done)]
    if limit is not None:
        pending = pending.head(limit)

    counts: dict = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for row in pending.itertuples(index=False):
            record = _fetch_one(session, row, delay)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            processed = sum(counts.values())
            if processed % progress_every == 0:
                print(f"{processed} of {len(pending)} done: {counts}", flush=True)
    return counts


def _fetch_one(session, row, delay: float) -> dict:
    """Fetch the page and PDF for one target and build its record.

    Args:
        session: A ``requests.Session`` with the project's user agent.
        row: One row of ``report_targets``.
        delay: Seconds to wait before each request.

    Returns:
        A JSON-ready record with ``status`` of "ok", "no_match",
        "page_error" or "pdf_error".
    """
    start = datetime.strptime(row.previous_inspection_start_date, "%d/%m/%Y").date()
    record = {
        "report_key": row.report_key,
        "urn": row.urn_at_time_of_previous_inspection,
        "previous_inspection_start_date": row.previous_inspection_start_date,
    }
    time.sleep(delay)
    try:
        page = session.get(PROVIDER_URL.format(urn=row.urn_at_time_of_previous_inspection), timeout=30)
        page.raise_for_status()
    except Exception as error:  # noqa: BLE001  network failures vary widely
        return {**record, "status": "page_error", "error": str(error)[:200]}

    links = parse_report_links(page.text)
    chosen, match = select_report(links, start)
    record.update(page_url=page.url, links_found=len(links), match=match)
    if chosen is None:
        return {**record, "status": "no_match"}

    time.sleep(delay)
    try:
        pdf = session.get(f"https://files.ofsted.gov.uk/v1/file/{chosen.file_id}", timeout=60)
        pdf.raise_for_status()
        pages = extract_pages(pdf.content)
    except Exception as error:  # noqa: BLE001  covers download and parsing failures
        failed = {"file_id": chosen.file_id, "listed_date": str(chosen.inspection_date), "kind": chosen.kind}
        return {**record, **failed, "status": "pdf_error", "error": str(error)[:200]}

    return {
        **record,
        "file_id": chosen.file_id,
        "listed_date": str(chosen.inspection_date),
        "kind": chosen.kind,
        "status": "ok",
        "n_pages": len(pages),
        "pages": pages,
    }
