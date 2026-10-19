"""Source files, year mappings and coding rules for the project.

Every year-specific decision lives here so that a reviewer can see which
source feeds which target year, and so the mappings can be tested rather
than buried inside loading code. File names are kept exactly as published,
because the data manifest records them that way.
"""

from pathlib import Path

RAW_DIR = Path("data/raw")

TARGET_YEARS = ("2021/22", "2022/23", "2023/24")
TRAIN_YEARS = ("2021/22", "2022/23")
TEST_YEAR = "2023/24"
DRIFT_YEAR = "2024/25"

BETWEEN_FILES = {
    "2021/22": "State_funded_schools_inspections_and_outcomes_between_1_September_2021_and_31_August_2022.csv",
    "2022/23": "State_funded_schools_inspections_and_outcomes_between_1_September_2022_and_31_August_2023.csv",
    "2023/24": "State_funded_schools_inspections_and_outcomes_between_1_September_2023_and_31_August_2024.csv",
    "2024/25": "State-funded_schools_inspections_and_outcomes_between_1_September_2024_and_31_August_2025.csv",
}

AS_AT_FILES = {
    "2021": "State_funded_schools_inspections_and_outcomes_as_at_31_August_2021.csv",
    "2022": "State_funded_schools_inspections_and_outcomes_as_at_31_August_2022.csv",
    "2023": "State_funded_schools_inspections_and_outcomes_as_at_31_August_2023.csv",
    "2024": "State_funded_schools_inspections_and_outcomes_as_at_31_August_2024.csv",
    "2025": "State-funded_schools_inspections_and_outcomes_as_at_31_August_2025.csv",
}

# Snapshot taken on 31 August just before each target year starts.
STANDING_SNAPSHOT = {"2021/22": "2021", "2022/23": "2022", "2023/24": "2023", "2024/25": "2024"}

# Snapshot published a year later, used to find late-published inspections.
COMPLETENESS_SNAPSHOT = {"2021/22": "2023", "2022/23": "2024", "2023/24": "2025"}

YEAR_WINDOWS = {
    "2021/22": ("2021-09-01", "2022-08-31"),
    "2022/23": ("2022-09-01", "2023-08-31"),
    "2023/24": ("2023-09-01", "2024-08-31"),
    "2024/25": ("2024-09-01", "2025-08-31"),
}

# Census files keyed by the target year they feed, with a one-year lag.
# The published names do not state the census month reliably, so each was
# identified from national headcount: January 2022 to January 2023 rose by
# 73,798 pupils, matching the 73,800 stated in the 2022/23 release.
CENSUS_FILES = {
    "2021/22": "spc_school_level_underlying_data_220216.csv",  # January 2021
    "2022/23": "spc_school_level_underlying_data_20230302.csv",  # January 2022
    "2023/24": "spc_school_level_underlying_data_23112023.csv",  # January 2023
    "2024/25": "spc_school_level_underlying_data.csv",  # January 2024
}

ABSENCE_FILE = Path("absence") / "data" / "1a_Absence_3term_school.csv"

# Absence for the academic year before each target year.
ABSENCE_PERIOD = {"2021/22": "202021", "2022/23": "202122", "2023/24": "202223", "2024/25": "202324"}

# These years count COVID non-attendance as possible sessions but not as
# absence, so their rates are not comparable with other years.
COVID_AFFECTED_PERIODS = ("202021", "202122")

GIAS_FILE = Path("gias") / "edubasealldata20261015.csv"
GIAS_LINKS_FILE = Path("gias") / "links_edubasealldata20261015.csv"

# A whitelist rather than a blacklist, so personal-data columns added to
# GIAS in future can never slip through.
GIAS_COLUMNS = (
    "URN",
    "EstablishmentName",
    "TypeOfEstablishment (name)",
    "EstablishmentStatus (name)",
    "ReasonEstablishmentOpened (name)",
    "OpenDate",
    "ReasonEstablishmentClosed (name)",
    "CloseDate",
    "PhaseOfEducation (name)",
    "UrbanRural (name)",
    "SchoolCapacity",
    "Trusts (code)",
)

OFSTED_ENCODING = "cp1252"

# Inspection types that give a comparable decline or no-decline outcome for
# a previously Good school. Monitoring and no-formal-designation visits are
# left out because they do not produce such a judgement.
INCLUDED_INSPECTION_TYPES = ("S5 Inspection", "Section 8 (did not convert)", "Section 8 conversion")

# Nursery schools have no compulsory-age absence data and PRUs follow
# different inspection dynamics, so both are out of the primary analysis.
EXCLUDED_PHASES = ("Nursery", "PRU")

STANDING_GRADE_GOOD = 2

CONCERNS_OUTCOME = "School remains Good (Concerns) - S5 Next"
NO_DECLINE_OUTCOMES = ("School remains Good", "School remains Good (Improving) - S5 Next")

# Tuples rather than sets, because pandas ``isin`` handles list-likes most
# consistently across versions.
VALID_GRADES = ("1", "2", "3", "4")
