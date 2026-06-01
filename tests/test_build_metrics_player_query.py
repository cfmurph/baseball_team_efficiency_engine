from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _load_fixture_tables(
    con: duckdb.DuckDBPyConnection,
    fact_player_season: pd.DataFrame,
    dim_team: pd.DataFrame,
    dim_player: pd.DataFrame,
) -> None:
    con.register("fact_player_season_df", fact_player_season)
    con.register("dim_team_df", dim_team)
    con.register("dim_player_df", dim_player)
    con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
    con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")
    con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")


def test_player_query_consolidates_traded_player_with_playing_time_weighted_rates() -> None:
    con = duckdb.connect(":memory:")
    try:
        _load_fixture_tables(
            con,
            fact_player_season=pd.DataFrame(
                [
                    {
                        "player_id": "traded",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 5,
                        "bb": 10,
                        "woba": 0.300,
                        "batting_war": 1.0,
                        "ip": 10.0,
                        "fip": 3.00,
                        "era": 4.00,
                        "pitching_war": 0.2,
                        "player_war": 1.2,
                        "salary": 1_000_000,
                        "surplus_value": 2_000_000,
                        "contract_label": "fair_value",
                    },
                    {
                        "player_id": "traded",
                        "season_key": 2024,
                        "team_id": "BOS",
                        "player_type": "both",
                        "pa": 300,
                        "hr": 20,
                        "bb": 40,
                        "woba": 0.400,
                        "batting_war": 3.0,
                        "ip": 30.0,
                        "fip": 5.00,
                        "era": 2.00,
                        "pitching_war": 0.8,
                        "player_war": 3.8,
                        "salary": 2_000_000,
                        "surplus_value": 6_000_000,
                        "contract_label": "surplus_value",
                    },
                ]
            ),
            dim_team=pd.DataFrame(
                [
                    {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_key": "BOS_2024", "team_id": "BOS", "team_name": "Boston Red Sox"},
                ]
            ),
            dim_player=pd.DataFrame(
                [
                    {
                        "player_id": "traded",
                        "name_full": "Traded Player",
                        "name_first": "Traded",
                        "name_last": "Player",
                    }
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["player_type"] == "both"
    assert row["pa"] == 400
    assert row["hr"] == 25
    assert row["bb"] == 50
    assert row["woba"] == pytest.approx(((0.300 * 100) + (0.400 * 300)) / 400)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(((3.00 * 10) + (5.00 * 30)) / 40)
    assert row["era"] == pytest.approx(((4.00 * 10) + (2.00 * 30)) / 40)
    assert row["player_war"] == pytest.approx(5.0)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 8_000_000
    assert row["contract_label"] == "surplus_value"


def test_player_query_joins_team_name_by_player_season_without_historical_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        _load_fixture_tables(
            con,
            fact_player_season=pd.DataFrame(
                [
                    {
                        "player_id": "current",
                        "season_key": 2024,
                        "team_id": "WSN",
                        "player_type": "batter",
                        "pa": 250,
                        "hr": 12,
                        "bb": 25,
                        "woba": 0.350,
                        "batting_war": 2.0,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 2.0,
                        "salary": 750_000,
                        "surplus_value": 5_000_000,
                        "contract_label": "surplus_value",
                    }
                ]
            ),
            dim_team=pd.DataFrame(
                [
                    {"team_key": "WSN_2004", "team_id": "WSN", "team_name": "Montreal Expos"},
                    {"team_key": "WSN_2024", "team_id": "WSN", "team_name": "Washington Nationals"},
                ]
            ),
            dim_player=pd.DataFrame(
                [
                    {
                        "player_id": "current",
                        "name_full": "Current National",
                        "name_first": "Current",
                        "name_last": "National",
                    }
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Washington Nationals"
    assert row["pa"] == 250
    assert row["player_war"] == pytest.approx(2.0)


def test_player_query_keeps_same_name_players_distinct_by_player_id() -> None:
    con = duckdb.connect(":memory:")
    try:
        _load_fixture_tables(
            con,
            fact_player_season=pd.DataFrame(
                [
                    {
                        "player_id": "same-a",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 4,
                        "bb": 8,
                        "woba": 0.310,
                        "batting_war": 0.8,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 0.8,
                        "salary": 500_000,
                        "surplus_value": 1_000_000,
                        "contract_label": "fair_value",
                    },
                    {
                        "player_id": "same-b",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 200,
                        "hr": 10,
                        "bb": 20,
                        "woba": 0.360,
                        "batting_war": 1.8,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 1.8,
                        "salary": 900_000,
                        "surplus_value": 3_000_000,
                        "contract_label": "surplus_value",
                    },
                ]
            ),
            dim_team=pd.DataFrame(
                [
                    {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
                ]
            ),
            dim_player=pd.DataFrame(
                [
                    {
                        "player_id": "same-a",
                        "name_full": "Chris Young",
                        "name_first": "Chris",
                        "name_last": "Young",
                    },
                    {
                        "player_id": "same-b",
                        "name_full": "Chris Young",
                        "name_first": "Chris",
                        "name_last": "Young",
                    },
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert set(result["player_id"]) == {"same-a", "same-b"}
    assert result["name_full"].tolist() == ["Chris Young", "Chris Young"]
