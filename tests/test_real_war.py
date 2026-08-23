"""Tests for Baseball-Reference rWAR overlay, ID mapping, and fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.baseball_analytics.war import (
    WAR_SOURCE_APPROX,
    WAR_SOURCE_MIXED,
    WAR_SOURCE_REAL,
    apply_real_war,
    load_br_team_map,
    load_real_war,
    map_br_player_ids,
    map_br_team_ids,
    team_war_from_players,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def team_map(tmp_path: Path) -> Path:
    src = Path("data/crosswalks/br_team_map.csv")
    dest = tmp_path / "br_team_map.csv"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


@pytest.fixture
def people() -> pd.DataFrame:
    # acunalu01 is the Lahman ID; BR uses acunajo01 (People.bbrefID)
    return pd.DataFrame({
        "playerID": ["judgeaa01", "troutmi01", "acunalu01", "unknown01"],
        "bbrefID": ["judgeaa01", "troutmi01", "acunajo01", np.nan],
    })


def test_map_br_player_ids_uses_bbref_crosswalk(people):
    war = pd.DataFrame({
        "player_ID": ["judgeaa01", "acunajo01"],
        "WAR": [10.8, 8.3],
    })
    mapped = map_br_player_ids(war, people)
    by_br = mapped.set_index("player_ID")["playerID"]
    assert by_br["judgeaa01"] == "judgeaa01"
    assert by_br["acunajo01"] == "acunalu01"


def test_map_br_player_ids_falls_back_to_player_id():
    people = pd.DataFrame({"playerID": ["a"], "bbrefID": ["a"]})
    war = pd.DataFrame({"player_ID": ["orphan99"], "WAR": [1.0]})
    mapped = map_br_player_ids(war, people)
    assert mapped.loc[0, "playerID"] == "orphan99"


def test_map_br_team_ids_year_aware(team_map):
    tm = load_br_team_map(team_map)
    df = pd.DataFrame({
        "team_ID": ["NYY", "MIL", "MIL", "TBD", "TBR", "WSN", "BOS"],
        "year_ID": [2022, 1995, 2011, 2004, 2015, 2019, 2010],
    })
    got = map_br_team_ids(df, tm).tolist()
    assert got == ["NYA", "ML4", "MIL", "TBA", "TBA", "WAS", "BOS"]


def test_apply_real_war_prefers_real():
    players = pd.DataFrame({
        "player_id": ["judgeaa01", "bench01"],
        "season_key": [2022, 2022],
        "team_id": ["NYA", "NYA"],
        "batting_war": [4.0, 1.5],
        "pitching_war": [0.0, 0.0],
        "player_war": [4.0, 1.5],
    })
    real = pd.DataFrame({
        "playerID": ["judgeaa01"],
        "yearID": [2022],
        "teamID": ["NYA"],
        "batting_war_real": [10.77],
        "pitching_war_real": [np.nan],
    })
    out = apply_real_war(players, real)
    judge = out.set_index("player_id").loc["judgeaa01"]
    bench = out.set_index("player_id").loc["bench01"]
    assert judge["batting_war"] == pytest.approx(10.77)
    assert judge["player_war"] == pytest.approx(10.77)
    assert judge["war_source"] == WAR_SOURCE_REAL
    assert bench["batting_war"] == pytest.approx(1.5)
    assert bench["war_source"] == WAR_SOURCE_APPROX


def test_apply_real_war_empty_keeps_approx():
    players = pd.DataFrame({
        "player_id": ["x"],
        "season_key": [2010],
        "team_id": ["BOS"],
        "batting_war": [2.0],
        "pitching_war": [0.5],
    })
    out = apply_real_war(players, pd.DataFrame())
    assert (out["war_source"] == WAR_SOURCE_APPROX).all()
    assert out.loc[0, "player_war"] == pytest.approx(2.5)


def test_apply_real_war_player_year_fallback_when_team_misses():
    """Unique player-year leftover attaches even if team IDs differ."""
    players = pd.DataFrame({
        "player_id": ["ohtansh01"],
        "season_key": [2023],
        "team_id": ["LAA"],
        "batting_war": [3.0],
        "pitching_war": [2.0],
    })
    real = pd.DataFrame({
        "playerID": ["ohtansh01"],
        "yearID": [2023],
        "teamID": ["ANA"],  # unmapped / wrong team id
        "batting_war_real": [6.11],
        "pitching_war_real": [3.80],
    })
    out = apply_real_war(players, real)
    assert out.loc[0, "war_source"] == WAR_SOURCE_REAL
    assert out.loc[0, "player_war"] == pytest.approx(9.91)


def test_apply_real_war_does_not_fallback_traded_player():
    """Multi-team player-years must not receive leftover WAR on every stint."""
    players = pd.DataFrame({
        "player_id": ["traded01", "traded01"],
        "season_key": [2015, 2015],
        "team_id": ["NYA", "BOS"],
        "batting_war": [1.0, 1.2],
        "pitching_war": [0.0, 0.0],
    })
    real = pd.DataFrame({
        "playerID": ["traded01"],
        "yearID": [2015],
        "teamID": ["NYY"],
        "batting_war_real": [4.0],
        "pitching_war_real": [np.nan],
    })
    out = apply_real_war(players, real)
    assert (out["war_source"] == WAR_SOURCE_APPROX).all()
    assert out["player_war"].tolist() == pytest.approx([1.0, 1.2])


def test_team_war_from_players_source_labels():
    players = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "season_key": [2010, 2010, 2010],
        "team_id": ["NYA", "NYA", "BOS"],
        "batting_war": [5.0, 1.0, 3.0],
        "pitching_war": [0.0, 0.0, 2.0],
        "player_war": [5.0, 1.0, 5.0],
        "war_source": [WAR_SOURCE_REAL, WAR_SOURCE_APPROX, WAR_SOURCE_REAL],
    })
    totals = team_war_from_players(players).set_index("teamID")
    assert totals.loc["NYA", "team_total_war"] == pytest.approx(6.0)
    assert totals.loc["NYA", "war_source"] == WAR_SOURCE_MIXED
    assert totals.loc["BOS", "war_source"] == WAR_SOURCE_REAL
    assert totals.loc["BOS", "team_pitching_war"] == pytest.approx(2.0)


def test_load_real_war_missing_files(tmp_path, people, team_map):
    result = load_real_war(tmp_path, people, min_year=1990, team_map_path=team_map)
    assert result.empty
    assert list(result.columns) == [
        "playerID", "yearID", "teamID", "batting_war_real", "pitching_war_real",
    ]


def test_load_real_war_maps_and_sums_stints(tmp_path, people, team_map):
    bat = pd.DataFrame({
        "name_common": ["Aaron Judge", "Ronald Acuna", "Ronald Acuna"],
        "player_ID": ["judgeaa01", "acunajo01", "acunajo01"],
        "year_ID": [2022, 2021, 2021],
        "team_ID": ["NYY", "ATL", "ATL"],
        "stint_ID": [1, 1, 2],
        "WAR": [10.77, 3.0, 2.5],
    })
    pit = pd.DataFrame({
        "name_common": ["Jacob deGrom"],
        "player_ID": ["degroja01"],
        "year_ID": [2018],
        "team_ID": ["NYM"],
        "stint_ID": [1],
        "WAR": [9.44],
    })
    bat.to_csv(tmp_path / "war_daily_bat.txt", index=False)
    pit.to_csv(tmp_path / "war_daily_pitch.txt", index=False)

    result = load_real_war(tmp_path, people, min_year=1990, team_map_path=team_map)
    keyed = result.set_index(["playerID", "yearID", "teamID"])
    assert keyed.loc[("judgeaa01", 2022, "NYA"), "batting_war_real"] == pytest.approx(10.77)
    assert keyed.loc[("acunalu01", 2021, "ATL"), "batting_war_real"] == pytest.approx(5.5)
    assert keyed.loc[("degroja01", 2018, "NYN"), "pitching_war_real"] == pytest.approx(9.44)
