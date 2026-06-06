from __future__ import annotations

import duckdb
import pandas as pd
import pytest

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


def test_player_query_aggregates_traded_player_without_historical_team_fanout() -> None:
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "traded01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 10.0,
                "hr": 1.0,
                "bb": 2.0,
                "woba": 0.500,
                "batting_war": 1.0,
                "ip": 10.0,
                "fip": 5.00,
                "era": 6.00,
                "pitching_war": 0.2,
                "player_war": 1.2,
                "salary": 10_000_000.0,
                "surplus_value": -400_000.0,
                "contract_label": "fair_value",
            },
            {
                "player_id": "traded01",
                "season_key": 2024,
                "team_id": "BOS",
                "player_type": "batter",
                "pa": 90.0,
                "hr": 9.0,
                "bb": 18.0,
                "woba": 0.300,
                "batting_war": 4.0,
                "ip": 90.0,
                "fip": 3.00,
                "era": 4.00,
                "pitching_war": 1.8,
                "player_war": 5.8,
                "salary": 20_000_000.0,
                "surplus_value": 1_400_000.0,
                "contract_label": "surplus_value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "traded01",
                "name_full": "Traded Player",
                "name_first": "Traded",
                "name_last": "Player",
            }
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
            {"team_key": "BOS_2024", "team_id": "BOS", "team_name": "Boston Red Sox"},
            {"team_key": "BOS_1908", "team_id": "BOS", "team_name": "Boston Americans"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == 100.0
    assert row["salary"] == 30_000_000.0
    assert row["surplus_value"] == 1_000_000.0
    assert row["player_war"] == pytest.approx(7.0)
    assert row["contract_label"] == "surplus_value"
    assert row["woba"] == pytest.approx(0.320)
    assert row["fip"] == pytest.approx(3.20)
    assert row["era"] == pytest.approx(4.20)


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "smithjo01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 200.0,
                "hr": 10.0,
                "bb": 20.0,
                "woba": 0.350,
                "batting_war": 2.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.0,
                "salary": 1_000_000.0,
                "surplus_value": 15_000_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "smithjo02",
                "season_key": 2024,
                "team_id": "BOS",
                "player_type": "pitcher",
                "pa": 0.0,
                "hr": 0.0,
                "bb": 0.0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 180.0,
                "fip": 3.25,
                "era": 3.50,
                "pitching_war": 4.0,
                "player_war": 4.0,
                "salary": 8_000_000.0,
                "surplus_value": 24_000_000.0,
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
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
            {"team_key": "BOS_2024", "team_id": "BOS", "team_name": "Boston Red Sox"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert set(result["player_id"]) == {"smithjo01", "smithjo02"}
    assert result["name_full"].tolist() == ["John Smith", "John Smith"]
    assert result.set_index("player_id").loc["smithjo01", "pa"] == 200.0
    assert result.set_index("player_id").loc["smithjo02", "ip"] == 180.0
