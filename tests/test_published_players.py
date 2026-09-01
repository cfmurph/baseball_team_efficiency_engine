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
    assert season["avg"] == pytest.approx(0.222)


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


def test_public_season_passes_fielding_and_omits_when_absent() -> None:
    with_fielding = {
        **JUDGE_2026,
        "putouts": "248",
        "assists": "7",
        "errors": "3",
        "double_plays": "2",
        "fielding_g": "112",
        "fielding_pos": "RF",
        "fielding_json": '[{"pos":"RF","g":112,"po":248,"a":7,"e":3,"dp":2,"fpct":0.988}]',
        "runs": "85",
        "doubles": "22",
        "triples": "1",
        "hr": "40",
        "sb": "8",
        "cs": "3",
        "hbp": "8",
        "sf": "5",
        "fielding_inn": "980",
    }
    season = public_player_season(with_fielding)
    assert season is not None
    assert season["runs"] == 85
    assert season["doubles"] == 22
    assert season["putouts"] == 248
    assert season["fpct"] == pytest.approx(0.988)
    assert season["cs"] == 3
    assert season["hbp"] == 8
    assert season["singles"] == 77
    assert season["xbh"] == 63
    assert season["tb"] == 284
    assert season["sb_pct"] == pytest.approx(8 / 11)
    assert season["tc"] == 258
    assert season["fielding"][0]["pos"] == "RF"
    assert season["fielding"][0]["po"] == 248
    assert season["fielding"][0]["tc"] == 258
    assert "dfs_salary" not in season
    assert "drs" not in season
    assert "uzr" not in season
    assert "oaa" not in season

    empty = public_player_season(JUDGE_2026)
    assert empty is not None
    assert empty["fielding"] == []
    assert empty["putouts"] is None
    assert empty["fpct"] is None
    assert empty.get("singles") is None
    assert empty.get("xbh") is None
    assert empty.get("tc") is None


def test_public_season_does_not_treat_batting_games_as_fielding() -> None:
    row = {
        **JUDGE_2026,
        "games": "120",
        "position": "OF",
    }
    season = public_player_season(row)
    assert season is not None
    assert season["games"] == 120
    assert season["fielding"] == []


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
