"""Header framing + data-health banner (dashboard SOP, Phase A). Pure functions, no model."""
from tools.dashboard_header import coverage_subtitle, data_health_note, format_header_title

PROFILE_NO_TIME = {"dataset": {"rows": 120, "columns": 5, "duplicate_rows": 0}, "columns": [
    {"name": "region", "role": "categorical", "missing_pct": 0},
    {"name": "revenue", "role": "numeric", "missing_pct": 0}], "time_column": None}

PROFILE_WITH_TIME = {"dataset": {"rows": 365, "columns": 4, "duplicate_rows": 0}, "columns": [
    {"name": "order_date", "role": "datetime", "min": "2026-01-01", "max": "2026-10-31", "missing_pct": 0},
    {"name": "revenue", "role": "numeric", "missing_pct": 0}], "time_column": "order_date"}


def test_format_header_title_adds_overview_suffix():
    assert format_header_title("Sales") == "Sales overview"


def test_format_header_title_does_not_double_up():
    assert format_header_title("Sales Dashboard") == "Sales Dashboard"
    assert format_header_title("Quarterly overview") == "Quarterly overview"


def test_format_header_title_handles_missing_title():
    assert format_header_title("") == "Dashboard"


def test_coverage_subtitle_uses_date_range_when_available():
    assert coverage_subtitle(PROFILE_WITH_TIME) == "Data from 2026-01-01 to 2026-10-31 · 365 rows"


def test_coverage_subtitle_falls_back_to_row_and_column_counts():
    assert coverage_subtitle(PROFILE_NO_TIME) == "120 rows across 5 columns"


def test_coverage_subtitle_handles_a_single_day_of_data():
    profile = {"dataset": {"rows": 50, "columns": 3}, "time_column": "order_date", "columns": [
        {"name": "order_date", "role": "datetime", "min": "2026-10-31", "max": "2026-10-31"}]}
    assert coverage_subtitle(profile) == "Data for 2026-10-31 · 50 rows"


def test_data_health_note_names_the_worst_missing_column():
    profile = {"dataset": {"rows": 100, "duplicate_rows": 0},
               "columns": [{"name": "revenue", "role": "numeric", "missing_pct": 8.0},
                           {"name": "region", "role": "categorical", "missing_pct": 2.0}]}
    assert data_health_note(profile) == "Data health: 'revenue' is missing 8.0% of rows."


def test_data_health_note_mentions_duplicates_alongside_missing_data():
    profile = {"dataset": {"rows": 100, "duplicate_rows": 3},
               "columns": [{"name": "revenue", "role": "numeric", "missing_pct": 8.0}]}
    note = data_health_note(profile)
    assert "missing 8.0%" in note and "3 duplicate rows found" in note


def test_data_health_note_is_positive_when_nothing_is_missing():
    profile = {"dataset": {"rows": 100, "duplicate_rows": 0},
               "columns": [{"name": "revenue", "role": "numeric", "missing_pct": 0.0}]}
    assert data_health_note(profile) == "Data health: no missing values found in this dataset."


def test_data_health_note_flags_duplicates_even_with_no_missing_data():
    profile = {"dataset": {"rows": 100, "duplicate_rows": 4},
               "columns": [{"name": "revenue", "role": "numeric", "missing_pct": 0.0}]}
    assert data_health_note(profile) == ("Data health: 4 duplicate rows found; "
                                         "all other columns are complete.")


def test_data_health_note_never_blames_id_or_empty_columns():
    profile = {"dataset": {"rows": 100, "duplicate_rows": 0},
               "columns": [{"name": "order_id", "role": "id", "missing_pct": 5.0},
                           {"name": "blank", "role": "empty", "missing_pct": 100.0}]}
    assert data_health_note(profile) == "Data health: no missing values found in this dataset."
