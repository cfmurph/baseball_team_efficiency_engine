"""Tests for warehouse table builders used by the dashboard pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.transform.build_warehouse import (
    build_dim_player,
    build_fact_player_season,
    build_fact_team_season,
)


def _batting_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "playerID": ["batter", "two_way"],
            "yearID": [2020, 2020],
            "teamID": ["AAA", "AAA"],
            "AB": [560, 420],
            "H": [180, 120],
            "2B": [35, 25],
            "3B": [4, 2],
            "HR": [32, 12],
            "BB": [70, 45],
            "IBB": [5, 1],
            "HBP": [6, 3],
            "SF": [5, 4],
            "SH": [0, 1],
        }
    )


def _pitching_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "playerID": ["pitcher", "two_way"],
            "yearID": [2020, 2020],
            "teamID": ["AAA", "AAA"],
            "IPouts": [0, 540],
            "HR": [0, 14],
            "BB": [0, 45],
            "HBP": [0, 4],
            "SO": [0, 165],
            "ERA": [np.nan, 3.35],
        }
    )


def _salary_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "yearID": [2020, 2020, 2020],
            "teamID": ["AAA", "AAA", "AAA"],
            "playerID": ["batter", "pitcher", "two_way"],
            "salary": [500_000, 7_000_000, 1_000_000],
        }
    )


def test_build_fact_player_season_classifies_player_types() -> None:
    result = build_fact_player_season(_batting_rows(), _pitching_rows(), _salary_rows(), min_year=2020)

    player_types = result.set_index("player_id")["player_type"].to_dict()

    assert player_types == {
        "batter": "batter",
        "pitcher": "pitcher",
        "two_way": "both",
    }


def test_build_fact_player_season_derives_contract_labels() -> None:
    result = build_fact_player_season(_batting_rows(), _pitching_rows(), _salary_rows(), min_year=2020)
    by_player = result.set_index("player_id")

    assert by_player.loc["batter", "player_war"] > 0
    assert by_player.loc["batter", "contract_label"] == "surplus_value"
    assert by_player.loc["pitcher", "player_war"] == 0
    assert by_player.loc["pitcher", "contract_label"] == "dead_money"


def test_build_fact_team_season_outputs_dashboard_metrics() -> None:
    teams = pd.DataFrame(
        {
            "team_key": ["AAA_2020"],
            "season_key": [2020],
            "yearID": [2020],
            "teamID": ["AAA"],
            "franchID": ["AAA"],
            "name": ["Analytics Aces"],
            "lgID": ["AL"],
            "W": [95],
            "L": [67],
            "G": [162],
            "R": [810],
            "RA": [650],
            "SOA": [1450],
            "attend": [2_500_000],
        }
    )

    result = build_fact_team_season(
        teams,
        _salary_rows(),
        _batting_rows(),
        _pitching_rows(),
        min_year=2020,
    )

    row = result.iloc[0]
    assert row["wins"] == 95
    assert row["team_total_war"] > 0
    assert np.isfinite(row["cost_per_war"])
    assert 0 <= row["dead_money_share"] <= 1
    assert row["window_phase"] == "contending"


def test_build_dim_player_handles_missing_optional_people_columns() -> None:
    people = pd.DataFrame(
        {
            "playerID": ["player1"],
            "nameFirst": ["Ada"],
            "nameLast": ["Lovelace"],
        }
    )

    result = build_dim_player(people)

    assert list(result.columns) == [
        "player_id",
        "name_first",
        "name_last",
        "name_full",
        "birth_year",
        "birth_country",
        "throws",
        "bats",
    ]
    assert result.loc[0, "name_full"] == "Ada Lovelace"
    assert pd.isna(result.loc[0, "birth_year"])
