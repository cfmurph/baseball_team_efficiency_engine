from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_collapses_stints_without_team_dimension_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register(
            "fact_player_season",
            pd.DataFrame(
                [
                    {
                        "player_id": "traded01",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 10,
                        "bb": 20,
                        "woba": 0.300,
                        "batting_war": 1.0,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 1.0,
                        "salary": 1_000_000.0,
                        "surplus_value": 7_000_000.0,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "traded01",
                        "season_key": 2024,
                        "team_id": "LAN",
                        "player_type": "batter",
                        "pa": 50,
                        "hr": 5,
                        "bb": 10,
                        "woba": 0.400,
                        "batting_war": 2.0,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": 2.0,
                        "salary": 2_000_000.0,
                        "surplus_value": 14_000_000.0,
                        "contract_label": "fair_value",
                    },
                    {
                        "player_id": "smithal01",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "pitcher",
                        "pa": 0,
                        "hr": 0,
                        "bb": 0,
                        "woba": None,
                        "batting_war": 0.0,
                        "ip": 120.0,
                        "fip": 3.50,
                        "era": 3.20,
                        "pitching_war": 3.0,
                        "player_war": 3.0,
                        "salary": 5_000_000.0,
                        "surplus_value": 19_000_000.0,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "smithal02",
                        "season_key": 2024,
                        "team_id": "NYA",
                        "player_type": "batter",
                        "pa": 80,
                        "hr": 2,
                        "bb": 8,
                        "woba": 0.250,
                        "batting_war": -0.5,
                        "ip": 0.0,
                        "fip": None,
                        "era": None,
                        "pitching_war": 0.0,
                        "player_war": -0.5,
                        "salary": 750_000.0,
                        "surplus_value": -4_750_000.0,
                        "contract_label": "dead_money",
                    },
                ]
            ),
        )
        con.register(
            "dim_player",
            pd.DataFrame(
                [
                    {"player_id": "traded01", "name_full": "Traded Player", "name_first": "Traded", "name_last": "Player"},
                    {"player_id": "smithal01", "name_full": "Alex Smith", "name_first": "Alex", "name_last": "Smith"},
                    {"player_id": "smithal02", "name_full": "Alex Smith", "name_first": "Alex", "name_last": "Smith"},
                ]
            ),
        )
        con.register(
            "dim_team",
            pd.DataFrame(
                [
                    {"team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_id": "NYA", "team_name": "New York Yankees"},
                    {"team_id": "LAN", "team_name": "Los Angeles Dodgers"},
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 3

    traded = result[result["player_id"] == "traded01"].iloc[0]
    assert traded["team_name"] == "Los Angeles Dodgers"
    assert traded["pa"] == pytest.approx(150)
    assert traded["hr"] == pytest.approx(15)
    assert traded["salary"] == pytest.approx(3_000_000.0)
    assert traded["player_war"] == pytest.approx(3.0)
    assert traded["contract_label"] == "fair_value"

    same_name = result[result["name_full"] == "Alex Smith"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["smithal01", "smithal02"]
