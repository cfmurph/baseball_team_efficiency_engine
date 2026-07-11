"""Regression tests for player metrics export SQL."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


def _insert_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    view_name = f"_test_{table}"
    con.register(view_name, df)
    columns = ", ".join(df.columns)
    con.execute(f"INSERT INTO {table} ({columns}) SELECT {columns} FROM {view_name}")
    con.unregister(view_name)


def _seed_dimensions(con: duckdb.DuckDBPyConnection) -> None:
    _insert_df(
        con,
        "dim_player",
        pd.DataFrame(
            [
                {
                    "player_id": "traded-player",
                    "name_first": "Shohei",
                    "name_last": "Example",
                    "name_full": "Shohei Example",
                },
                {
                    "player_id": "same-name-a",
                    "name_first": "Chris",
                    "name_last": "Young",
                    "name_full": "Chris Young",
                },
                {
                    "player_id": "same-name-b",
                    "name_first": "Chris",
                    "name_last": "Young",
                    "name_full": "Chris Young",
                },
            ]
        ),
    )
    _insert_df(
        con,
        "dim_team",
        pd.DataFrame(
            [
                {"team_key": "OAK_2023", "team_id": "OAK", "team_name": "Oakland Athletics"},
                {"team_key": "OAK_2024", "team_id": "OAK", "team_name": "Oakland Athletics"},
                {"team_key": "BOS_2024", "team_id": "BOS", "team_name": "Boston Red Sox"},
                {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
            ]
        ),
    )


def test_player_query_collapses_traded_stints_without_team_dimension_fanout() -> None:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    _seed_dimensions(con)
    _insert_df(
        con,
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "traded-player",
                    "season_key": 2024,
                    "team_id": "OAK",
                    "player_type": "batter",
                    "pa": 100,
                    "hr": 5,
                    "bb": 10,
                    "woba": 0.320,
                    "batting_war": 0.7,
                    "ip": None,
                    "fip": None,
                    "era": None,
                    "pitching_war": 0.0,
                    "player_war": 0.7,
                    "salary": 1_000_000.0,
                    "surplus_value": 4_600_000.0,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "traded-player",
                    "season_key": 2024,
                    "team_id": "BOS",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 20.0,
                    "fip": 3.20,
                    "era": 3.60,
                    "pitching_war": 1.2,
                    "player_war": 1.2,
                    "salary": 2_000_000.0,
                    "surplus_value": 7_600_000.0,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "traded-player",
                    "season_key": 2024,
                    "team_id": "NYA",
                    "player_type": "both",
                    "pa": 25,
                    "hr": 2,
                    "bb": 3,
                    "woba": 0.410,
                    "batting_war": 0.3,
                    "ip": 10.0,
                    "fip": 2.80,
                    "era": 2.70,
                    "pitching_war": 1.8,
                    "player_war": 2.1,
                    "salary": 3_000_000.0,
                    "surplus_value": 13_800_000.0,
                    "contract_label": "fair_value",
                },
            ]
        ),
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    player_rows = result[result["player_id"] == "traded-player"]
    assert len(player_rows) == 1
    row = player_rows.iloc[0]
    assert row["year_id"] == 2024
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "New York Yankees"
    assert row["player_type"] == "both"
    assert row["pa"] == pytest.approx(125)
    assert row["hr"] == pytest.approx(7)
    assert row["bb"] == pytest.approx(13)
    assert row["ip"] == pytest.approx(30)
    assert row["batting_war"] == pytest.approx(1.0)
    assert row["pitching_war"] == pytest.approx(3.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(6_000_000.0)
    assert row["contract_label"] == "fair_value"
    assert not result.duplicated(["player_id", "year_id"]).any()

    con.close()


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    _seed_dimensions(con)
    _insert_df(
        con,
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "same-name-a",
                    "season_key": 2024,
                    "team_id": "BOS",
                    "player_type": "batter",
                    "pa": 400,
                    "hr": 12,
                    "bb": 40,
                    "woba": 0.330,
                    "batting_war": 2.0,
                    "pitching_war": 0.0,
                    "player_war": 2.0,
                    "salary": 5_000_000.0,
                    "surplus_value": 11_000_000.0,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "same-name-b",
                    "season_key": 2024,
                    "team_id": "NYA",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "batting_war": 0.0,
                    "ip": 120.0,
                    "fip": 3.70,
                    "era": 4.00,
                    "pitching_war": 1.5,
                    "player_war": 1.5,
                    "salary": 4_000_000.0,
                    "surplus_value": 8_000_000.0,
                    "contract_label": "surplus_value",
                },
            ]
        ),
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    same_name = result[result["name_full"] == "Chris Young"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["same-name-a", "same-name-b"]
    assert same_name["team_id"].tolist() == ["BOS", "NYA"]
    assert same_name["player_war"].tolist() == pytest.approx([2.0, 1.5])

    con.close()
