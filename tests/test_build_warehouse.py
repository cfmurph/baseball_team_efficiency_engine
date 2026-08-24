from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.transform import build_warehouse
from pipeline.transform.build_warehouse import (
    build_dim_player,
    build_fact_player_season,
    build_fact_team_season,
)

pytestmark = pytest.mark.integration


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


def _teams_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "team_key": ["LAA_2020"],
        "season_key": [2020],
        "yearID": [2020],
        "teamID": ["LAA"],
        "W": [85],
        "L": [77],
        "G": [162],
        "R": [750],
        "RA": [720],
        "SOA": [1300],
        "attend": [3_000_000],
    })


def _salaries_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "yearID": [2020],
        "teamID": ["LAA"],
        "playerID": ["twoway01"],
        "salary": [5_000_000.0],
    })


def _batting_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "playerID": ["twoway01"],
        "yearID": [2020],
        "teamID": ["LAA"],
        "AB": [200],
        "H": [50],
        "X2B": [10],
        "X3B": [1],
        "HR": [5],
        "BB": [20],
        "HBP": [2],
        "SF": [3],
        "SH": [0],
        "IBB": [1],
    })


def _pitching_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "playerID": ["twoway01"],
        "yearID": [2020],
        "teamID": ["LAA"],
        "IPouts": [300],
        "HR": [10],
        "BB": [30],
        "HBP": [4],
        "SO": [120],
        "ERA": [3.50],
    })


def _stub_two_way_war(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        build_warehouse,
        "batting_war",
        lambda _batting: pd.DataFrame({
            "playerID": ["twoway01"],
            "yearID": [2020],
            "teamID": ["LAA"],
            "batting_war": [0.2],
            "pa": [225],
            "woba": [0.310],
            "hr": [5],
        }),
    )
    monkeypatch.setattr(
        build_warehouse,
        "pitching_war",
        lambda _pitching: pd.DataFrame({
            "playerID": ["twoway01"],
            "yearID": [2020],
            "teamID": ["LAA"],
            "pitching_war": [1.0],
            "ip": [100.0],
            "fip": [3.20],
            "era": [3.50],
        }),
    )


def test_build_fact_player_season_merges_two_way_war_and_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_two_way_war(monkeypatch)

    result = build_warehouse.build_fact_player_season(
        _batting_frame(),
        _pitching_frame(),
        _salaries_frame(),
        min_year=2020,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_type"] == "both"
    assert row["player_war"] == pytest.approx(1.2)
    assert row["contract_label"] == "surplus_value"


def test_build_fact_team_season_dead_money_share_counts_two_way_player_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_two_way_war(monkeypatch)

    player_season = build_warehouse.build_fact_player_season(
        _batting_frame(),
        _pitching_frame(),
        _salaries_frame(),
        min_year=2020,
    )
    result = build_warehouse.build_fact_team_season(
        _teams_frame(),
        _salaries_frame(),
        _batting_frame(),
        _pitching_frame(),
        min_year=2020,
        player_season=player_season,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_total_war"] == pytest.approx(1.2)
    assert row["payroll"] == pytest.approx(5_000_000.0)
    assert row["dead_money_share"] == pytest.approx(0.0)

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

