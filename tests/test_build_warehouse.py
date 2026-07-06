from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transform.build_warehouse import build_fact_player_season


def test_build_fact_player_season_derives_player_types_and_contract_labels():
    batting = pd.DataFrame(
        [
            {
                "playerID": "bat-only",
                "yearID": 2020,
                "teamID": "NYA",
                "AB": 500,
                "H": 150,
                "2B": 30,
                "3B": 2,
                "HR": 20,
                "BB": 55,
                "IBB": 4,
                "HBP": 3,
                "SF": 5,
                "SH": 1,
            },
            {
                "playerID": "two-way",
                "yearID": 2020,
                "teamID": "LAA",
                "AB": 400,
                "H": 120,
                "2B": 25,
                "3B": 4,
                "HR": 15,
                "BB": 45,
                "IBB": 2,
                "HBP": 2,
                "SF": 4,
                "SH": 0,
            },
        ]
    )
    pitching = pd.DataFrame(
        [
            {
                "playerID": "pitch-only",
                "yearID": 2020,
                "teamID": "BOS",
                "IPouts": 540,
                "HR": 18,
                "BB": 45,
                "HBP": 4,
                "SO": 170,
                "ERA": 3.80,
            },
            {
                "playerID": "two-way",
                "yearID": 2020,
                "teamID": "LAA",
                "IPouts": 300,
                "HR": 10,
                "BB": 35,
                "HBP": 3,
                "SO": 110,
                "ERA": 3.50,
            },
        ]
    )
    salaries = pd.DataFrame(
        [
            {"playerID": "bat-only", "yearID": 2020, "teamID": "NYA", "salary": 1_000_000.0},
            {"playerID": "pitch-only", "yearID": 2020, "teamID": "BOS", "salary": 2_000_000.0},
            {"playerID": "two-way", "yearID": 2020, "teamID": "LAA", "salary": 3_000_000.0},
        ]
    )

    result = build_fact_player_season(batting, pitching, salaries, min_year=2020)
    by_player = result.set_index("player_id")

    assert by_player.loc["bat-only", "player_type"] == "batter"
    assert by_player.loc["pitch-only", "player_type"] == "pitcher"
    assert by_player.loc["two-way", "player_type"] == "both"
    assert by_player.loc["bat-only", "salary"] == pytest.approx(1_000_000.0)
    assert by_player.loc["pitch-only", "salary"] == pytest.approx(2_000_000.0)
    assert by_player.loc["two-way", "salary"] == pytest.approx(3_000_000.0)
    assert by_player["contract_label"].notna().all()
    assert set(by_player["contract_label"]).issubset(
        {"dead_money", "overpaid", "fair_value", "surplus_value", "unknown"}
    )
    assert by_player["player_war"].equals(by_player["batting_war"] + by_player["pitching_war"])
