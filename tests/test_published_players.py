"""Unit tests for published player_season_metrics projection."""
from __future__ import annotations

import pytest

from src.baseball_analytics.published import (
    group_public_players,
    public_player_season,
    resolve_published_player,
)

pytestmark = pytest.mark.unit


JUDGE_2026 = {
    "player_id": "judgeaa01",
    "player_name": "Aaron Judge",
    "season": "2026",
    "team": "NYY",
    "team_name": "Yankees",
    "position": "OF",
    "player_type": "batter",
    "pa": "500",
    "hits": "140",
    "ab": "400",
    "hr": "40",
    "war_source": "real",
    "player_war": "6.1",
    "vs_replacement": "3.4",
    "salary": "40000000",
    "surplus_value": "1",
    "stat_source": "sportsdataio",
    "dfs_salary": "9800",
    "betting_line": "-150",
}

JUDGE_2024 = {
    **JUDGE_2026,
    "season": "2024",
    "player_war": "10.8",
    "stat_source": "lahman",
}

TROUT_2023 = {
    "player_id": "troutmi01",
    "player_name": "Mike Trout",
    "season": "2023",
    "team": "LAA",
    "position": "OF",
    "pa": "410",
    "hits": "90",
    "ab": "350",
    "player_war": "9.0",
    "war_source": "real",
}


def test_public_season_strips_vs_repl_payroll_and_vendor_dumps() -> None:
    season = public_player_season(JUDGE_2026)
    assert season is not None
    assert season["season"] == 2026
    assert season["pa"] == 500
    assert season["hits"] == 140
    assert season["avg"] == pytest.approx(0.35)
    assert season["war"] == pytest.approx(6.1)
    assert season["stat_source"] == "sportsdataio"
    dumped = str(season)
    assert "vs_replacement" not in season
    assert "vs repl" not in dumped
    assert "salary" not in season
    assert "surplus_value" not in season
    assert "dfs_salary" not in season
    assert "betting_line" not in season
    assert 40000000 not in season.values()


def test_public_season_does_not_invent_war() -> None:
    row = {
        "player_id": "x01",
        "player_name": "No War",
        "season": "2026",
        "pa": "20",
        "hits": "4",
        "ab": "18",
        "stat_source": "sportsdataio",
    }
    season = public_player_season(row)
    assert season is not None
    assert season["war"] is None
    assert season["war_source"] is None
    assert season["avg"] == pytest.approx(4 / 18)


def test_group_default_window_drops_years_outside_y_minus_2() -> None:
    players = group_public_players(
        [JUDGE_2026, JUDGE_2024, TROUT_2023],
        window=[2024, 2025, 2026],
    )
    ids = [item["player_id"] for item in players]
    assert ids == ["judgeaa01"]
    assert [row["season"] for row in players[0]["seasons"]] == [2026, 2024]


def test_resolve_unknown_player_is_none() -> None:
    assert resolve_published_player([JUDGE_2026], "nope") is None


def test_resolve_known_player_empty_seasons_when_year_missing() -> None:
    resolved = resolve_published_player(
        [JUDGE_2024],
        "judgeaa01",
        season=2026,
    )
    assert resolved is not None
    assert resolved["player_id"] == "judgeaa01"
    assert resolved["name"] == "Aaron Judge"
    assert resolved["seasons"] == []
