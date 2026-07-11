"""Tests for warehouse DataFrame builders."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transform.build_warehouse import build_fact_player_season


def test_build_fact_player_season_classifies_types_and_merges_salary() -> None:
    batting = pd.DataFrame(
        [
            {
                "playerID": "batter-only",
                "yearID": 2010,
                "teamID": "NYA",
                "AB": 520,
                "H": 160,
                "X2B": 35,
                "X3B": 4,
                "HR": 30,
                "BB": 60,
                "IBB": 5,
                "HBP": 4,
                "SF": 3,
                "SH": 1,
            },
            {
                "playerID": "two-way",
                "yearID": 2010,
                "teamID": "NYA",
                "AB": 150,
                "H": 45,
                "X2B": 8,
                "X3B": 1,
                "HR": 5,
                "BB": 20,
                "IBB": 1,
                "HBP": 2,
                "SF": 1,
                "SH": 0,
            },
        ]
    )
    pitching = pd.DataFrame(
        [
            {
                "playerID": "pitcher-only",
                "yearID": 2010,
                "teamID": "NYA",
                "IPouts": 600,
                "HR": 18,
                "BB": 55,
                "HBP": 6,
                "SO": 180,
                "ERA": 3.50,
            },
            {
                "playerID": "two-way",
                "yearID": 2010,
                "teamID": "NYA",
                "IPouts": 90,
                "HR": 2,
                "BB": 8,
                "HBP": 1,
                "SO": 28,
                "ERA": 2.90,
            },
        ]
    )
    salaries = pd.DataFrame(
        [
            {"playerID": "batter-only", "yearID": 2010, "teamID": "NYA", "salary": 8_000_000.0},
            {"playerID": "pitcher-only", "yearID": 2010, "teamID": "NYA", "salary": 10_000_000.0},
            {"playerID": "two-way", "yearID": 2010, "teamID": "NYA", "salary": 750_000.0},
        ]
    )

    result = build_fact_player_season(batting, pitching, salaries, min_year=2010)
    by_player = result.set_index("player_id")

    assert by_player.loc["batter-only", "player_type"] == "batter"
    assert by_player.loc["pitcher-only", "player_type"] == "pitcher"
    assert by_player.loc["two-way", "player_type"] == "both"
    assert by_player.loc["batter-only", "salary"] == pytest.approx(8_000_000.0)
    assert by_player.loc["pitcher-only", "salary"] == pytest.approx(10_000_000.0)
    assert by_player.loc["two-way", "salary"] == pytest.approx(750_000.0)

    for player_id in ["batter-only", "pitcher-only", "two-way"]:
        row = by_player.loc[player_id]
        assert row["player_war"] == pytest.approx(row["batting_war"] + row["pitching_war"])
        assert isinstance(row["contract_label"], str)
        assert row["contract_label"]

    assert pd.isna(by_player.loc["batter-only", "ip"])
    assert pd.isna(by_player.loc["pitcher-only", "pa"])
