from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_collapses_traded_player_stints_without_team_fanout() -> None:
    """The exported player metrics should remain one row per player-season."""
    con = duckdb.connect(":memory:")
    con.register(
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "player-a",
                    "season_key": 2024,
                    "team_id": "AAA",
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
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "player-a",
                    "season_key": 2024,
                    "team_id": "BBB",
                    "player_type": "batter",
                    "pa": 300,
                    "hr": 15,
                    "bb": 30,
                    "woba": 0.400,
                    "batting_war": 3.0,
                    "ip": None,
                    "fip": None,
                    "era": None,
                    "pitching_war": 0.0,
                    "player_war": 3.0,
                    "salary": 2_000_000,
                    "surplus_value": 22_000_000,
                    "contract_label": "fair_value",
                },
            ]
        ),
    )
    con.register(
        "dim_player",
        pd.DataFrame(
            [
                {
                    "player_id": "player-a",
                    "name_full": "Traded Batter",
                    "name_first": "Traded",
                    "name_last": "Batter",
                }
            ]
        ),
    )
    con.register(
        "dim_team",
        pd.DataFrame(
            [
                {"team_id": "AAA", "team_name": "Alpha Aces"},
                {"team_id": "AAA", "team_name": "Alpha Aces"},
                {"team_id": "BBB", "team_name": "Beta Bears"},
                {"team_id": "BBB", "team_name": "Beta Bears"},
            ]
        ),
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-a"
    assert row["year_id"] == 2024
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Bears"
    assert row["pa"] == 400
    assert row["hr"] == 20
    assert row["bb"] == 40
    assert row["batting_war"] == pytest.approx(4.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 29_000_000
    assert row["woba"] == pytest.approx(0.375)
    assert row["contract_label"] == "fair_value"


def test_player_query_weights_pitching_rate_stats_and_preserves_two_way_type() -> None:
    con = duckdb.connect(":memory:")
    con.register(
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "player-b",
                    "season_key": 2024,
                    "team_id": "AAA",
                    "player_type": "both",
                    "pa": 10,
                    "hr": 1,
                    "bb": 2,
                    "woba": 0.500,
                    "batting_war": 0.5,
                    "ip": 10.0,
                    "fip": 5.00,
                    "era": 6.00,
                    "pitching_war": 1.5,
                    "player_war": 2.0,
                    "salary": 4_000_000,
                    "surplus_value": 12_000_000,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "player-b",
                    "season_key": 2024,
                    "team_id": "BBB",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 30.0,
                    "fip": 3.00,
                    "era": 2.00,
                    "pitching_war": 1.0,
                    "player_war": 1.0,
                    "salary": 1_000_000,
                    "surplus_value": 7_000_000,
                    "contract_label": "overpaid",
                },
            ]
        ),
    )
    con.register(
        "dim_player",
        pd.DataFrame(
            [
                {
                    "player_id": "player-b",
                    "name_full": "Two Way Pitcher",
                    "name_first": "Two Way",
                    "name_last": "Pitcher",
                }
            ]
        ),
    )
    con.register(
        "dim_team",
        pd.DataFrame(
            [
                {"team_id": "AAA", "team_name": "Alpha Aces"},
                {"team_id": "BBB", "team_name": "Beta Bears"},
            ]
        ),
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "AAA"
    assert row["player_type"] == "both"
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(3.5)
    assert row["era"] == pytest.approx(3.0)
    assert row["pitching_war"] == pytest.approx(2.5)
    assert row["player_war"] == pytest.approx(3.0)
    assert row["contract_label"] == "surplus_value"
