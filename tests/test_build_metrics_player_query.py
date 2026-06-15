from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _run_player_query(
    fact_player_season: pd.DataFrame,
    dim_player: pd.DataFrame,
    dim_team: pd.DataFrame,
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.register("fact_player_season", fact_player_season)
        con.register("dim_player", dim_player)
        con.register("dim_team", dim_team)
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def test_player_query_keeps_same_name_players_distinct_by_id() -> None:
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "smithjo01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 400,
                "hr": 20,
                "bb": 40,
                "woba": 0.350,
                "batting_war": 3.0,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 3.0,
                "salary": 5_000_000,
                "surplus_value": 19_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "smithjo02",
                "season_key": 2024,
                "team_id": "BOS",
                "player_type": "pitcher",
                "pa": None,
                "hr": None,
                "bb": None,
                "woba": None,
                "batting_war": 0.0,
                "ip": 170.0,
                "fip": 3.25,
                "era": 3.40,
                "pitching_war": 4.0,
                "player_war": 4.0,
                "salary": 10_000_000,
                "surplus_value": 22_000_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {"player_id": "smithjo01", "name_full": "John Smith", "name_first": "John", "name_last": "Smith"},
            {"player_id": "smithjo02", "name_full": "John Smith", "name_first": "John", "name_last": "Smith"},
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_id": "NYA", "team_name": "New York Yankees"},
            {"team_id": "BOS", "team_name": "Boston Red Sox"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert len(result) == 2
    assert set(result["player_id"]) == {"smithjo01", "smithjo02"}
    assert (result["name_full"] == "John Smith").all()


def test_player_query_collapses_traded_player_without_team_dimension_fanout() -> None:
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "traded01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 200,
                "hr": 8,
                "bb": 20,
                "woba": 0.320,
                "batting_war": 1.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "salary": 2_000_000,
                "surplus_value": 10_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "traded01",
                "season_key": 2024,
                "team_id": "BOS",
                "player_type": "batter",
                "pa": 300,
                "hr": 12,
                "bb": 35,
                "woba": 0.360,
                "batting_war": 2.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.5,
                "salary": 3_000_000,
                "surplus_value": 17_000_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [{"player_id": "traded01", "name_full": "Trade Target", "name_first": "Trade", "name_last": "Target"}]
    )
    dim_team = pd.DataFrame(
        [
            {"team_id": "NYA", "team_name": "New York Yankees"},
            {"team_id": "NYA", "team_name": "New York Yankees"},
            {"team_id": "BOS", "team_name": "Boston Red Sox"},
            {"team_id": "BOS", "team_name": "Boston Red Sox"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded01"
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == 500
    assert row["hr"] == 20
    assert row["bb"] == 55
    assert row["batting_war"] == 4.0
    assert row["player_war"] == 4.0
    assert row["salary"] == 5_000_000
    assert row["surplus_value"] == 27_000_000
