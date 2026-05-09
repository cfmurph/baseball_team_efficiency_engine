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
        con.register("fact_player_season_df", fact_player_season)
        con.register("dim_player_df", dim_player)
        con.register("dim_team_df", dim_team)
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _fact_rows(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "player_id",
        "season_key",
        "team_id",
        "player_type",
        "pa",
        "hr",
        "bb",
        "woba",
        "batting_war",
        "ip",
        "fip",
        "era",
        "pitching_war",
        "player_war",
        "salary",
        "surplus_value",
        "contract_label",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_player_query_aggregates_traded_player_with_weighted_rates() -> None:
    fact_player_season = _fact_rows(
        [
            {
                "player_id": "traded-player",
                "season_key": 2020,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 100,
                "hr": 5,
                "bb": 10,
                "woba": 0.400,
                "batting_war": 1.0,
                "ip": 10.0,
                "fip": 4.00,
                "era": 4.50,
                "pitching_war": 0.2,
                "player_war": 1.2,
                "salary": 1_000_000,
                "surplus_value": 2_000_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "traded-player",
                "season_key": 2020,
                "team_id": "BBB",
                "player_type": "pitcher",
                "pa": 300,
                "hr": 15,
                "bb": 20,
                "woba": 0.300,
                "batting_war": 2.0,
                "ip": 30.0,
                "fip": 3.00,
                "era": 2.50,
                "pitching_war": 1.8,
                "player_war": 3.8,
                "salary": 3_000_000,
                "surplus_value": 6_000_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "traded-player",
                "name_full": "Traded Player",
                "name_first": "Traded",
                "name_last": "Player",
            }
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "AAA_2020", "team_id": "AAA", "team_name": "Alpha"},
            {"team_key": "BBB_2020", "team_id": "BBB", "team_name": "Beta"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert len(result) == 1
    player = result.iloc[0]
    assert player["team_id"] == "BBB"
    assert player["team_name"] == "Beta"
    assert player["player_type"] == "pitcher"
    assert player["pa"] == 400
    assert player["hr"] == 20
    assert player["bb"] == 30
    assert player["woba"] == pytest.approx(0.325)
    assert player["ip"] == pytest.approx(40.0)
    assert player["fip"] == pytest.approx(3.25)
    assert player["era"] == pytest.approx(3.0)
    assert player["player_war"] == pytest.approx(5.0)
    assert player["salary"] == 4_000_000
    assert player["surplus_value"] == 8_000_000
    assert player["contract_label"] == "surplus_value"


def test_player_query_joins_dim_team_by_player_season_without_fanout() -> None:
    fact_player_season = _fact_rows(
        [
            {
                "player_id": "single-season",
                "season_key": 2020,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 120,
                "hr": 8,
                "bb": 12,
                "woba": 0.350,
                "batting_war": 1.5,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "salary": 2_000_000,
                "surplus_value": 4_000_000,
                "contract_label": "surplus_value",
            }
        ]
    )
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "single-season",
                "name_full": "Single Season",
                "name_first": "Single",
                "name_last": "Season",
            }
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "AAA_2019", "team_id": "AAA", "team_name": "Old Alpha"},
            {"team_key": "AAA_2020", "team_id": "AAA", "team_name": "Current Alpha"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert len(result) == 1
    player = result.iloc[0]
    assert player["team_name"] == "Current Alpha"
    assert player["pa"] == 120
    assert player["salary"] == 2_000_000


def test_player_query_keeps_same_name_players_distinct_by_player_id() -> None:
    fact_player_season = _fact_rows(
        [
            {
                "player_id": "j-smith-1",
                "season_key": 2021,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 200,
                "hr": 10,
                "bb": 20,
                "woba": 0.330,
                "batting_war": 2.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.0,
                "salary": 1_000_000,
                "surplus_value": 5_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "j-smith-2",
                "season_key": 2021,
                "team_id": "BBB",
                "player_type": "pitcher",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 50.0,
                "fip": 3.10,
                "era": 3.20,
                "pitching_war": 1.5,
                "player_war": 1.5,
                "salary": 1_500_000,
                "surplus_value": 3_000_000,
                "contract_label": "fair_value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "j-smith-1",
                "name_full": "Jordan Smith",
                "name_first": "Jordan",
                "name_last": "Smith",
            },
            {
                "player_id": "j-smith-2",
                "name_full": "Jordan Smith",
                "name_first": "Jordan",
                "name_last": "Smith",
            },
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "AAA_2021", "team_id": "AAA", "team_name": "Alpha"},
            {"team_key": "BBB_2021", "team_id": "BBB", "team_name": "Beta"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert set(result["player_id"]) == {"j-smith-1", "j-smith-2"}
    assert result["name_full"].tolist().count("Jordan Smith") == 2
