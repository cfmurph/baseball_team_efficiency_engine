"""Regression tests for dashboard metric export SQL."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY, _table_has_rows


def _load_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    con.register(f"{name}_df", df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM {name}_df")
    con.unregister(f"{name}_df")


@pytest.fixture
def player_metrics_connection():
    con = duckdb.connect(":memory:")
    _load_table(
        con,
        "dim_player",
        pd.DataFrame(
            [
                {
                    "player_id": "traded01",
                    "name_full": "Taylor Traded",
                    "name_first": "Taylor",
                    "name_last": "Traded",
                },
                {
                    "player_id": "single01",
                    "name_full": "Sam Single",
                    "name_first": "Sam",
                    "name_last": "Single",
                },
            ]
        ),
    )
    _load_table(
        con,
        "dim_team",
        pd.DataFrame(
            [
                {"team_id": "NYY", "team_name": "New York Yankees"},
                {"team_id": "NYY", "team_name": "New York Yankees"},
                {"team_id": "BOS", "team_name": "Boston Red Sox"},
            ]
        ),
    )
    _load_table(
        con,
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "traded01",
                    "season_key": 2024,
                    "team_id": "NYY",
                    "player_type": "batter",
                    "pa": 300,
                    "hr": 10,
                    "bb": 30,
                    "woba": 0.340,
                    "batting_war": 1.5,
                    "ip": 0.0,
                    "fip": None,
                    "era": None,
                    "pitching_war": 0.0,
                    "player_war": 1.5,
                    "salary": 3_000_000,
                    "surplus_value": 9_000_000,
                    "contract_label": "surplus",
                },
                {
                    "player_id": "traded01",
                    "season_key": 2024,
                    "team_id": "BOS",
                    "player_type": "batter",
                    "pa": 200,
                    "hr": 8,
                    "bb": 20,
                    "woba": 0.360,
                    "batting_war": 2.0,
                    "ip": 0.0,
                    "fip": None,
                    "era": None,
                    "pitching_war": 0.0,
                    "player_war": 2.0,
                    "salary": 2_000_000,
                    "surplus_value": 11_000_000,
                    "contract_label": "elite_value",
                },
                {
                    "player_id": "single01",
                    "season_key": 2024,
                    "team_id": "NYY",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 150.0,
                    "fip": 3.5,
                    "era": 3.2,
                    "pitching_war": 4.0,
                    "player_war": 4.0,
                    "salary": 12_000_000,
                    "surplus_value": 20_000_000,
                    "contract_label": "surplus",
                },
            ]
        ),
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_aggregates_traded_player_to_one_season_row(player_metrics_connection):
    result = player_metrics_connection.execute(_PLAYER_QUERY).fetchdf()

    traded = result[result["player_id"] == "traded01"]
    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["year_id"] == 2024
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == 500
    assert row["hr"] == 18
    assert row["player_war"] == pytest.approx(3.5)
    assert row["salary"] == 5_000_000
    assert row["surplus_value"] == 20_000_000
    assert row["contract_label"] == "elite_value"


def test_player_query_is_not_fanned_out_by_duplicate_team_dimension_rows(player_metrics_connection):
    result = player_metrics_connection.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 2
    assert not result.duplicated(["player_id", "year_id"]).any()


def test_table_has_rows_handles_missing_empty_and_populated_tables():
    con = duckdb.connect(":memory:")
    try:
        assert not _table_has_rows(con, "missing_table")

        con.execute("CREATE TABLE empty_table (id INTEGER)")
        assert not _table_has_rows(con, "empty_table")

        con.execute("CREATE TABLE populated_table AS SELECT 1 AS id")
        assert _table_has_rows(con, "populated_table")
    finally:
        con.close()
