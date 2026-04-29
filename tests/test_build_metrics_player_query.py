from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_consolidates_traded_player_without_team_history_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register(
            "dim_player",
            pd.DataFrame(
                [
                    {
                        "player_id": "traded01",
                        "name_full": "Traded Player",
                        "name_first": "Traded",
                        "name_last": "Player",
                    },
                    {
                        "player_id": "single01",
                        "name_full": "Single Team",
                        "name_first": "Single",
                        "name_last": "Team",
                    },
                ]
            ),
        )
        con.register(
            "dim_team",
            pd.DataFrame(
                [
                    {"team_key": "NYA_1999", "season_key": 1999, "team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_key": "NYA_2000", "season_key": 2000, "team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_key": "ANA_2000", "season_key": 2000, "team_id": "ANA", "team_name": "Anaheim Angels"},
                    {"team_key": "ANA_2005", "season_key": 2005, "team_id": "ANA", "team_name": "Los Angeles Angels"},
                ]
            ),
        )
        con.register(
            "fact_player_season",
            pd.DataFrame(
                [
                    {
                        "player_id": "traded01",
                        "season_key": 2000,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 10,
                        "bb": 20,
                        "woba": 0.400,
                        "batting_war": 2.0,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 2.0,
                        "salary": 1_000_000,
                        "surplus_value": 13_000_000,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "traded01",
                        "season_key": 2000,
                        "team_id": "ANA",
                        "player_type": "batter",
                        "pa": 50,
                        "hr": 5,
                        "bb": 10,
                        "woba": 0.300,
                        "batting_war": 1.0,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 1.0,
                        "salary": 500_000,
                        "surplus_value": 6_500_000,
                        "contract_label": "fair_value",
                    },
                    {
                        "player_id": "single01",
                        "season_key": 2000,
                        "team_id": "ANA",
                        "player_type": "pitcher",
                        "pa": 0,
                        "hr": 0,
                        "bb": 0,
                        "woba": None,
                        "batting_war": 0.0,
                        "ip": 80.0,
                        "fip": 3.25,
                        "era": 3.50,
                        "pitching_war": 2.5,
                        "player_war": 2.5,
                        "salary": 750_000,
                        "surplus_value": 16_750_000,
                        "contract_label": "surplus_value",
                    },
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert result[["player_id", "year_id"]].duplicated().sum() == 0
    assert len(result) == 2

    traded = result[result["player_id"] == "traded01"].iloc[0]
    assert traded["team_id"] == "NYA"
    assert traded["team_name"] == "New York Yankees"
    assert traded["pa"] == 150
    assert traded["hr"] == 15
    assert traded["bb"] == 30
    assert traded["salary"] == 1_500_000
    assert traded["player_war"] == 3.0
    assert traded["contract_label"] == "surplus_value"

    single_team = result[result["player_id"] == "single01"].iloc[0]
    assert single_team["team_name"] == "Anaheim Angels"
    assert single_team["ip"] == 80.0
    assert single_team["player_war"] == 2.5
