from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_collapses_traded_stints_without_dim_team_fanout():
    con = duckdb.connect(":memory:")
    try:
        con.register(
            "fact_player_season_df",
            pd.DataFrame(
                [
                    {
                        "player_id": "player-a",
                        "season_key": 2020,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 10,
                        "bb": 15,
                        "woba": 0.410,
                        "batting_war": 2.5,
                        "ip": None,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 2.5,
                        "salary": 5_000_000.0,
                        "surplus_value": 15_000_000.0,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "player-a",
                        "season_key": 2020,
                        "team_id": "BOS",
                        "player_type": "pitcher",
                        "pa": None,
                        "hr": None,
                        "bb": None,
                        "woba": None,
                        "batting_war": 0.0,
                        "ip": 45.0,
                        "fip": 3.50,
                        "era": 3.20,
                        "pitching_war": 1.0,
                        "player_war": 1.0,
                        "salary": 2_000_000.0,
                        "surplus_value": 6_000_000.0,
                        "contract_label": "fair_value",
                    },
                ]
            ),
        )
        con.register(
            "dim_player_df",
            pd.DataFrame(
                [
                    {
                        "player_id": "player-a",
                        "name_full": "Alex Sample",
                        "name_first": "Alex",
                        "name_last": "Sample",
                    }
                ]
            ),
        )
        con.register(
            "dim_team_df",
            pd.DataFrame(
                [
                    {
                        "team_key": "NYA_2019",
                        "team_id": "NYA",
                        "team_name": "New York Yankees",
                    },
                    {
                        "team_key": "NYA_2020",
                        "team_id": "NYA",
                        "team_name": "New York Yankees",
                    },
                    {
                        "team_key": "BOS_2020",
                        "team_id": "BOS",
                        "team_name": "Boston Red Sox",
                    },
                ]
            ),
        )
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-a"
    assert row["year_id"] == 2020
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "New York Yankees"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 100
    assert row["ip"] == pytest.approx(45.0)
    assert row["player_war"] == pytest.approx(3.5)
    assert row["salary"] == pytest.approx(7_000_000.0)
    assert row["surplus_value"] == pytest.approx(21_000_000.0)
    assert row["contract_label"] == "surplus_value"


def test_player_query_preserves_same_name_players_by_player_id():
    con = duckdb.connect(":memory:")
    try:
        con.register(
            "fact_player_season_df",
            pd.DataFrame(
                [
                    {
                        "player_id": "smith-1",
                        "season_key": 2021,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 300,
                        "hr": 12,
                        "bb": 30,
                        "woba": 0.340,
                        "batting_war": 1.8,
                        "ip": None,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 1.8,
                        "salary": 1_000_000.0,
                        "surplus_value": 13_400_000.0,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "smith-2",
                        "season_key": 2021,
                        "team_id": "BOS",
                        "player_type": "batter",
                        "pa": 120,
                        "hr": 3,
                        "bb": 10,
                        "woba": 0.300,
                        "batting_war": 0.4,
                        "ip": None,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 0.4,
                        "salary": 750_000.0,
                        "surplus_value": 2_450_000.0,
                        "contract_label": "surplus_value",
                    },
                ]
            ),
        )
        con.register(
            "dim_player_df",
            pd.DataFrame(
                [
                    {
                        "player_id": "smith-1",
                        "name_full": "Chris Smith",
                        "name_first": "Chris",
                        "name_last": "Smith",
                    },
                    {
                        "player_id": "smith-2",
                        "name_full": "Chris Smith",
                        "name_first": "Chris",
                        "name_last": "Smith",
                    },
                ]
            ),
        )
        con.register(
            "dim_team_df",
            pd.DataFrame(
                [
                    {"team_key": "NYA_2021", "team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_key": "BOS_2021", "team_id": "BOS", "team_name": "Boston Red Sox"},
                ]
            ),
        )
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 2
    assert set(result["player_id"]) == {"smith-1", "smith-2"}
    assert result["name_full"].tolist() == ["Chris Smith", "Chris Smith"]
