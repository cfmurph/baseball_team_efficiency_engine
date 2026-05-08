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


def _fact_rows() -> list[dict]:
    return [
        {
            "player_id": "traded-batter",
            "season_key": 2024,
            "team_id": "OAK",
            "player_type": "batter",
            "pa": 100,
            "hr": 5,
            "bb": 10,
            "woba": 0.300,
            "batting_war": 1.0,
            "ip": None,
            "fip": None,
            "era": None,
            "pitching_war": 0.0,
            "player_war": 1.0,
            "salary": 1_000_000,
            "surplus_value": 7_000_000,
            "contract_label": "value",
        },
        {
            "player_id": "traded-batter",
            "season_key": 2024,
            "team_id": "NYA",
            "player_type": "batter",
            "pa": 300,
            "hr": 15,
            "bb": 40,
            "woba": 0.400,
            "batting_war": 3.0,
            "ip": None,
            "fip": None,
            "era": None,
            "pitching_war": 0.0,
            "player_war": 3.0,
            "salary": 3_000_000,
            "surplus_value": 21_000_000,
            "contract_label": "surplus_star",
        },
        {
            "player_id": "traded-pitcher",
            "season_key": 2024,
            "team_id": "OAK",
            "player_type": "pitcher",
            "pa": None,
            "hr": None,
            "bb": None,
            "woba": None,
            "batting_war": 0.0,
            "ip": 50.0,
            "fip": 2.00,
            "era": 3.00,
            "pitching_war": 2.0,
            "player_war": 2.0,
            "salary": 2_000_000,
            "surplus_value": 14_000_000,
            "contract_label": "surplus_star",
        },
        {
            "player_id": "traded-pitcher",
            "season_key": 2024,
            "team_id": "NYA",
            "player_type": "pitcher",
            "pa": None,
            "hr": None,
            "bb": None,
            "woba": None,
            "batting_war": 0.0,
            "ip": 150.0,
            "fip": 4.00,
            "era": 5.00,
            "pitching_war": 1.0,
            "player_war": 1.0,
            "salary": 4_000_000,
            "surplus_value": 7_000_000,
            "contract_label": "value",
        },
    ]


def test_player_query_aggregates_traded_players_without_team_dimension_fanout() -> None:
    fact_player_season = pd.DataFrame(_fact_rows())
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "traded-batter",
                "name_full": "Pat Transfer",
                "name_first": "Pat",
                "name_last": "Transfer",
            },
            {
                "player_id": "traded-pitcher",
                "name_full": "Casey Cutter",
                "name_first": "Casey",
                "name_last": "Cutter",
            },
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "OAK_2024", "team_id": "OAK", "team_name": "Athletics"},
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "Yankees"},
            # Historical name for the same team_id should not duplicate current-season facts.
            {"team_key": "NYA_2023", "team_id": "NYA", "team_name": "New York Yankees"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    assert set(result["player_id"]) == {"traded-batter", "traded-pitcher"}
    batter = result.set_index("player_id").loc["traded-batter"]
    assert batter["team_name"] == "Yankees"
    assert batter["pa"] == 400
    assert batter["hr"] == 20
    assert batter["salary"] == 4_000_000
    assert batter["woba"] == pytest.approx(0.375)

    pitcher = result.set_index("player_id").loc["traded-pitcher"]
    assert pitcher["team_name"] == "Athletics"
    assert pitcher["ip"] == pytest.approx(200.0)
    assert pitcher["salary"] == 6_000_000
    assert pitcher["fip"] == pytest.approx(3.50)
    assert pitcher["era"] == pytest.approx(4.50)


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "smith-a",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 200,
                "hr": 8,
                "bb": 20,
                "woba": 0.330,
                "batting_war": 1.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "salary": 1_500_000,
                "surplus_value": 10_500_000,
                "contract_label": "value",
            },
            {
                "player_id": "smith-b",
                "season_key": 2024,
                "team_id": "OAK",
                "player_type": "batter",
                "pa": 150,
                "hr": 3,
                "bb": 30,
                "woba": 0.310,
                "batting_war": 0.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.5,
                "salary": 800_000,
                "surplus_value": 3_500_000,
                "contract_label": "value",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {"player_id": "smith-a", "name_full": "Alex Smith", "name_first": "Alex", "name_last": "Smith"},
            {"player_id": "smith-b", "name_full": "Alex Smith", "name_first": "Alex", "name_last": "Smith"},
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "Yankees"},
            {"team_key": "OAK_2024", "team_id": "OAK", "team_name": "Athletics"},
        ]
    )

    result = _run_player_query(fact_player_season, dim_player, dim_team)

    alex_rows = result[result["name_full"] == "Alex Smith"].sort_values("player_id")
    assert alex_rows["player_id"].tolist() == ["smith-a", "smith-b"]
    assert alex_rows["pa"].tolist() == [200, 150]
