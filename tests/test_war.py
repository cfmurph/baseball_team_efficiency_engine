"""Tests for the WAR approximation module."""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from src.baseball_analytics.war import (
    batting_war,
    pitching_war,
    team_war_totals,
    base_runs,
)


# ---- Fixtures ----

@pytest.fixture
def sample_batting() -> pd.DataFrame:
    return pd.DataFrame({
        "playerID": ["playerA", "playerB", "playerC"],
        "yearID": [2010, 2010, 2010],
        "teamID": ["NYA", "NYA", "BOS"],
        "AB": [550, 500, 480],
        "H": [160, 130, 145],
        "2B": [30, 25, 28],
        "3B": [4, 2, 3],
        "HR": [25, 10, 20],
        "BB": [60, 50, 55],
        "IBB": [5, 3, 4],
        "HBP": [5, 3, 6],
        "SF": [4, 3, 5],
        "SH": [1, 2, 0],
    })


@pytest.fixture
def sample_pitching() -> pd.DataFrame:
    return pd.DataFrame({
        "playerID": ["pitcherA", "pitcherB"],
        "yearID": [2010, 2010],
        "teamID": ["NYA", "BOS"],
        "IPouts": [600, 540],   # 200 IP, 180 IP
        "HR": [20, 15],
        "BB": [55, 48],
        "HBP": [8, 6],
        "SO": [180, 160],
        "ERA": [3.50, 4.10],
    })


# ---- Tests ----

def test_batting_war_returns_dataframe(sample_batting):
    result = batting_war(sample_batting)
    assert isinstance(result, pd.DataFrame)
    assert "batting_war" in result.columns
    assert len(result) > 0


def test_batting_war_positive_for_good_hitters(sample_batting):
    result = batting_war(sample_batting)
    # Players with above-average stats should have positive WAR
    assert (result["batting_war"] > 0).all(), "Expected all players to have positive batting WAR"


def test_batting_war_pa_column(sample_batting):
    result = batting_war(sample_batting)
    assert "pa" in result.columns
    assert (result["pa"] > 0).all()


def test_pitching_war_returns_dataframe(sample_pitching):
    result = pitching_war(sample_pitching)
    assert isinstance(result, pd.DataFrame)
    assert "pitching_war" in result.columns


def test_pitching_war_fip_computed(sample_pitching):
    result = pitching_war(sample_pitching)
    assert "fip" in result.columns
    assert result["fip"].notna().all()


def test_team_war_totals(sample_batting, sample_pitching):
    bat = batting_war(sample_batting)
    pit = pitching_war(sample_pitching)
    totals = team_war_totals(bat, pit)
    assert "team_total_war" in totals.columns
    assert len(totals) >= 2  # NYA and BOS
    assert (totals["team_total_war"] > 0).all()


def test_team_war_totals_columns(sample_batting, sample_pitching):
    bat = batting_war(sample_batting)
    pit = pitching_war(sample_pitching)
    totals = team_war_totals(bat, pit)
    assert "team_batting_war" in totals.columns
    assert "team_pitching_war" in totals.columns


def test_base_runs_positive():
    hits = pd.Series([1500])
    singles = pd.Series([900])
    doubles = pd.Series([300])
    triples = pd.Series([50])
    hr = pd.Series([200])
    bb = pd.Series([450])
    hbp = pd.Series([60])
    ab = pd.Series([5500])
    sf = pd.Series([50])
    result = base_runs(hits, singles, doubles, triples, hr, bb, hbp, ab, sf)
    assert float(result.iloc[0]) > 0


def test_base_runs_zero_ab():
    hits = pd.Series([0])
    singles = pd.Series([0])
    doubles = pd.Series([0])
    triples = pd.Series([0])
    hr = pd.Series([0])
    bb = pd.Series([0])
    hbp = pd.Series([0])
    ab = pd.Series([0])
    sf = pd.Series([0])
    result = base_runs(hits, singles, doubles, triples, hr, bb, hbp, ab, sf)
    # With zero outs, denominator → 0, should be nan
    assert np.isnan(float(result.iloc[0])) or float(result.iloc[0]) == 0


def test_batting_war_handles_missing_columns():
    """Should not raise even if optional columns are absent."""
    minimal = pd.DataFrame({
        "playerID": ["A"],
        "yearID": [2010],
        "teamID": ["NYA"],
        "AB": [500],
        "H": [140],
        "2B": [25],
        "3B": [3],
        "HR": [18],
        "BB": [50],
    })
    result = batting_war(minimal)
    assert "batting_war" in result.columns
