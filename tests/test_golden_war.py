"""Golden rWAR seasons — fixtures only, no live Baseball-Reference download."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.baseball_analytics.war import (
    WAR_SOURCE_REAL,
    apply_real_war,
    load_real_war,
)

pytestmark = pytest.mark.e2e

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "war"
TEAM_MAP = Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "br_team_map.csv"
EXPECTED_PATH = FIXTURE_DIR / "expected.json"


def _require_fixtures() -> list[dict]:
    missing = [
        name
        for name in ("war_daily_bat.txt", "war_daily_pitch.txt", "people.csv", "expected.json")
        if not (FIXTURE_DIR / name).is_file()
    ]
    if missing:
        pytest.fail(f"Golden WAR fixtures missing: {missing}. See docs/war_sources.md.")
    return json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))["seasons"]


def test_golden_war_known_seasons_are_real() -> None:
    seasons = _require_fixtures()
    people = pd.read_csv(FIXTURE_DIR / "people.csv")
    real = load_real_war(FIXTURE_DIR, people, min_year=1990, team_map_path=TEAM_MAP)
    assert not real.empty

    players = pd.DataFrame(
        {
            "player_id": [row["player_id"] for row in seasons],
            "season_key": [row["year_id"] for row in seasons],
            "team_id": [row["team_id"] for row in seasons],
            "batting_war": [0.0] * len(seasons),
            "pitching_war": [0.0] * len(seasons),
        }
    )
    out = apply_real_war(players, real).set_index(["player_id", "season_key"])

    for row in seasons:
        got = out.loc[(row["player_id"], row["year_id"])]
        assert got["war_source"] == row["war_source"] == WAR_SOURCE_REAL, row["name"]
        assert got["player_war"] == pytest.approx(row["player_war"], abs=0.05), row["name"]
        if row["batting_war"] is not None:
            assert got["batting_war"] == pytest.approx(row["batting_war"], abs=0.05), row["name"]
        if row["pitching_war"] is not None:
            assert got["pitching_war"] == pytest.approx(row["pitching_war"], abs=0.05), row["name"]
