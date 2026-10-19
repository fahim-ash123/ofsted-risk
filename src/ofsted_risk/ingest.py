"""Loading and normalising the raw Ofsted, DfE and GIAS files.

The published files vary between years in encoding, header wording and
missing-value markers. This module absorbs those differences so that every
later step works with one consistent set of snake_case column names and
proper missing values. Each loader records what it cleaned in the frame's
``attrs`` so the data-quality report can state it rather than hide it.
"""

import re
from pathlib import Path

import pandas as pd

from ofsted_risk import config

_WARNING_NOTICES = re.compile(r"number of warning notices issued in \d{4}/\d{2} academic year", re.IGNORECASE)

# Older releases used different wording for the same field. Mapping them
# here keeps the rest of the pipeline free of year-specific branches.
HEADER_ALIASES = {
    "section_8_inspection_of_a_good_and_outstanding_school_overall_outcome": "ungraded_inspection_overall_outcome",
    "date_of_latest_section_8_inspection_of_a_good_and_outstanding_school": "date_of_latest_ungraded_inspection",
    "number_of_section_8_inspection_of_a_good_and_outstanding_schools_since_the_last_full_inspection": (
        "number_of_ungraded_inspections_since_the_last_graded_inspection"
    ),
    "does_the_section_8_inspection_of_a_good_and_outstanding_school_relate_to_the_urn_of_the_current_school": (
        "does_the_ungraded_inspection_relate_to_the_urn_of_the_current_school"
    ),
    "inspection_number_of_latest_full_inspection": "inspection_number_of_latest_graded_inspection",
    "latest_section_8_inspection_of_a_good_and_outstanding_school_number_since_last_full_inspection": (
        "latest_ungraded_inspection_number_since_last_graded_inspection"
    ),
    "headcount_total_girls": "headcount_total_female",
    "headcount_total_boys": "headcount_total_male",
}

MISSING_MARKERS = ("", "NULL")


def normalise_header(name: str) -> str:
    """Convert a published column header into a stable snake_case name.

    Args:
        name: Header exactly as it appears in the source file.

    Returns:
        A lower-case, underscore-separated name, with known year-specific
        variants mapped to one canonical form.
    """
    text = name.replace("(opens in new window)", "")
    text = _WARNING_NOTICES.sub("number of warning notices issued in academic year", text)
    text = text.replace("%", " pct ").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return HEADER_ALIASES.get(text, text)


def read_ofsted_csv(path: Path) -> pd.DataFrame:
    """Read an Ofsted inspections CSV, dropping blank rows and missing markers.

    Every column is read as text first. Grades such as 9 are codes rather
    than numbers, and parsing them early would hide that distinction.

    Args:
        path: Location of an Ofsted "as at" or "between" CSV.

    Returns:
        A frame of nullable strings with normalised headers, the literal
        strings "NULL" and "" replaced by missing values, and wholly blank
        rows removed. The
        number of blank rows dropped is stored in
        ``attrs["blank_rows_dropped"]``.

    Raises:
        KeyError: If the file has no URN column after normalisation.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=config.OFSTED_ENCODING)
    frame.columns = [normalise_header(column) for column in frame.columns]
    if "urn" not in frame.columns:
        raise KeyError(f"No URN column in {path}")
    blank = frame["urn"].str.strip() == ""
    # The nullable string type keeps comparisons against missing values
    # well defined in both pandas 2 and pandas 3.
    frame = frame.loc[~blank].replace(list(MISSING_MARKERS), pd.NA).astype("string").reset_index(drop=True)
    frame.attrs["blank_rows_dropped"] = int(blank.sum())
    frame.attrs["source"] = Path(path).name
    return frame


def parse_grade(series: pd.Series) -> pd.Series:
    """Turn a judgement column into integer grades 1 to 4.

    Args:
        series: Text judgement column from an Ofsted file.

    Returns:
        A nullable integer series. Codes that are not grades (9 for not
        applicable, 0, "Not judged" and missing values) become missing.
    """
    kept = series.where(series.isin(config.VALID_GRADES))
    return pd.to_numeric(kept, errors="coerce").astype("Int64")


def non_grade_codes(series: pd.Series) -> dict:
    """Count the values in a judgement column that are not grades 1 to 4.

    Args:
        series: Text judgement column from an Ofsted file.

    Returns:
        A mapping from each non-grade value to its count, with missing
        values reported under "<missing>".
    """
    other = series[~series.isin(config.VALID_GRADES)].fillna("<missing>")
    return other.value_counts().to_dict()


def parse_ofsted_date(series: pd.Series) -> pd.Series:
    """Parse Ofsted day-first dates.

    Args:
        series: Text dates in dd/mm/yyyy form.

    Returns:
        A datetime series, with anything unparseable left missing.
    """
    return pd.to_datetime(series, format="%d/%m/%Y", errors="coerce")


def load_between(year: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load the inspections carried out in one academic year.

    Args:
        year: Academic year such as "2023/24".
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per inspection, as returned by ``read_ofsted_csv``.
    """
    return read_ofsted_csv(raw_dir / config.BETWEEN_FILES[year])


def load_as_at(snapshot: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load the most-recent-inspection snapshot for 31 August of one year.

    Args:
        snapshot: Calendar year of the snapshot, such as "2023".
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per school, as returned by ``read_ofsted_csv``.
    """
    return read_ofsted_csv(raw_dir / config.AS_AT_FILES[snapshot])


def _read_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV as UTF-8, falling back to Windows-1252.

    DfE files are not consistent: some census years are UTF-8 and others
    Windows-1252, and nothing in the file name says which.

    Args:
        path: CSV to read.
        **kwargs: Passed on to ``pandas.read_csv``.

    Returns:
        The loaded frame, with the encoding used in ``attrs["encoding"]``.
    """
    for encoding in ("utf-8", "cp1252"):
        try:
            frame = pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
        frame.attrs["encoding"] = encoding
        return frame
    raise UnicodeDecodeError("utf-8/cp1252", b"", 0, 1, f"Could not decode {path}")


def load_census(year: str, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load school-level census context for the census feeding a target year.

    Args:
        year: Target academic year, such as "2023/24". The census loaded is
            the January before that year starts.
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per school with ``urn``, ``census_headcount``,
        ``census_pct_fsm`` and ``census_pct_eal``. Non-numeric tokens found
        in the numeric columns are counted in ``attrs["non_numeric"]``.
    """
    frame = _read_with_fallback(raw_dir / config.CENSUS_FILES[year], dtype=str, keep_default_na=False)
    frame.columns = [normalise_header(column) for column in frame.columns]
    columns = {
        "headcount_of_pupils": "census_headcount",
        "pct_of_pupils_known_to_be_eligible_for_free_school_meals": "census_pct_fsm",
        "pct_of_pupils_whose_first_language_is_known_or_believed_to_be_other_than_english": "census_pct_eal",
    }
    selected = frame[["urn", *columns]].rename(columns=columns)
    non_numeric = {}
    for column in columns.values():
        numeric = pd.to_numeric(selected[column], errors="coerce")
        tokens = selected.loc[numeric.isna(), column]
        non_numeric[column] = tokens.value_counts().to_dict()
        selected[column] = numeric
    selected.attrs.update(frame.attrs, non_numeric=non_numeric, source=config.CENSUS_FILES[year])
    return selected


def load_absence(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load school-level full-year absence for every published year.

    Args:
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per school and academic year, with ``urn``,
        ``time_period``, ``education_phase``, ``absence_enrolments``,
        ``absence_overall_pct``, ``absence_persistent_pct`` and
        ``absence_covid_sessions``.
    """
    columns = {
        "school_urn": "urn",
        "time_period": "time_period",
        "education_phase": "education_phase",
        "enrolments": "absence_enrolments",
        "sess_overall_percent": "absence_overall_pct",
        "enrolments_pa_10_exact_percent": "absence_persistent_pct",
        "sess_x_covid": "absence_covid_sessions",
    }
    frame = pd.read_csv(raw_dir / config.ABSENCE_FILE, dtype=str, keep_default_na=False, usecols=list(columns))
    frame = frame.rename(columns=columns)
    for column in ("absence_enrolments", "absence_overall_pct", "absence_persistent_pct", "absence_covid_sessions"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_gias(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load the GIAS establishment register without personal data.

    Only whitelisted columns are read, so headteacher names and contact
    details never enter the pipeline. GIAS is a current snapshot, so only
    attributes that change rarely, such as urban or rural setting, should
    be used as features.

    Args:
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per establishment, open or closed, with normalised headers.
    """
    frame = pd.read_csv(
        raw_dir / config.GIAS_FILE,
        dtype=str,
        keep_default_na=False,
        encoding=config.OFSTED_ENCODING,
        usecols=list(config.GIAS_COLUMNS),
    )
    frame.columns = [normalise_header(column) for column in frame.columns]
    return frame.replace("", pd.NA).astype("string")


def load_gias_links(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """Load GIAS predecessor and successor links between URNs.

    Args:
        raw_dir: Folder holding the raw downloads.

    Returns:
        One row per link, with normalised headers.
    """
    frame = pd.read_csv(raw_dir / config.GIAS_LINKS_FILE, dtype=str, keep_default_na=False, encoding=config.OFSTED_ENCODING)
    frame.columns = [normalise_header(column) for column in frame.columns]
    return frame
