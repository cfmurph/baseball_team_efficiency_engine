"""Tests for metric export helpers in build_metrics."""
from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.transform.build_metrics import (
    _dead_money_leaders,
    _efficiency_labels,
    _table_has_rows,
    _window_summary,
)


def test_efficiency_labels_assigns_expected_bins() -> None:
    df = pd.DataFrame({"wins_per_10m": [0.25, 0.75, 1.25, 2.0]})

    result = _efficiency_labels(df)

    assert result["efficiency_label"].astype(str).tolist() == [
        "low",
        "below_avg",
        "above_avg",
        "elite",
    ]


def test_window_summary_uses_latest_team_season() -> None:
    team_df = pd.DataFrame(
        {
            "team_name": ["Aces", "Aces", "Bees", "Bees"],
            "year_id": [2021, 2020, 2020, 2022],
            "window_phase": ["contending", "rebuilding", "steady", "developing"],
            "wins": [91, 65, 80, 86],
            "payroll": [120_000_000, 55_000_000, 75_000_000, 90_000_000],
            "team_total_war": [42.0, 18.0, 29.0, 35.0],
        }
    )

    result = _window_summary(team_df).set_index("team_name")

    assert result.loc["Aces", "year_id"] == 2021
    assert result.loc["Aces", "window_phase"] == "contending"
    assert result.loc["Bees", "year_id"] == 2022
    assert result.loc["Bees", "window_phase"] == "developing"


def test_dead_money_leaders_filters_and_sorts_by_salary() -> None:
    player_df = pd.DataFrame(
        {
            "name_full": ["Low Salary", "Healthy Contract", "High Salary"],
            "contract_label": ["dead_money", "surplus_value", "dead_money"],
            "salary": [4_000_000, 30_000_000, 25_000_000],
        }
    )

    result = _dead_money_leaders(player_df)

    assert result["name_full"].tolist() == ["High Salary", "Low Salary"]
    assert (result["contract_label"] == "dead_money").all()


def test_table_has_rows_handles_missing_empty_and_populated_tables() -> None:
    con = duckdb.connect(":memory:")
    try:
        assert not _table_has_rows(con, "missing_table")

        con.execute("CREATE TABLE optional_feed (id INTEGER)")
        assert not _table_has_rows(con, "optional_feed")

        con.execute("INSERT INTO optional_feed VALUES (1)")
        assert _table_has_rows(con, "optional_feed")
    finally:
        con.close()
