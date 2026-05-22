from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transform import build_warehouse


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

    result = build_warehouse.build_fact_team_season(
        _teams_frame(),
        _salaries_frame(),
        _batting_frame(),
        _pitching_frame(),
        min_year=2020,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_total_war"] == pytest.approx(1.2)
    assert row["payroll"] == pytest.approx(5_000_000.0)
    assert row["dead_money_share"] == pytest.approx(0.0)
