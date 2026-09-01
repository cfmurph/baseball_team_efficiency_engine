"""SDIO → player/team metrics overlay and current-season coverage (#131, #138)."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import (
    METRICS_MANIFEST_NAME,
    _SDIO_PLAYER_GAME_ROLLUP_QUERY,
    _SDIO_PLAYER_SEASON_QUERY,
    _SDIO_TEAM_GAME_ROLLUP_QUERY,
    _SDIO_TEAM_SEASON_QUERY,
    _SDIO_TEAM_STANDINGS_QUERY,
    approx_vs_replacement_from_counting,
    attach_team_coverage,
    bridge_sdio_player_season_metrics,
    bridge_sdio_team_season_metrics,
    enrich_player_season_phase0,
    write_metrics_manifest,
)
from dashboard.helpers import years_from_frame
from src.baseball_analytics.fantasy import rank_fantasy_cards
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.sportsdataio import (
    default_season_window,
    load_sdio_frames,
    seasons_from_settings,
    write_raw_payload,
)
from src.baseball_analytics.storage import decide_current_promote
from pipeline.transform.build_warehouse import insert_sdio_spine_tables

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).parent / "fixtures" / "sportsdataio"
AS_OF = "2026-08-23"


def _payload(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _lahman_frame(*years: int) -> pd.DataFrame:
    rows = []
    for year in years:
        rows.append(
            {
                "player_id": "judgeaa01",
                "name_full": "Aaron Judge",
                "name_first": "Aaron",
                "name_last": "Judge",
                "year_id": year,
                "team_id": "NYA",
                "team_name": "New York Yankees",
                "player_type": "batter",
                "pa": 700,
                "hr": 50,
                "bb": 120,
                "woba": 0.420,
                "batting_war": 10.8,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 10.8,
                "war_source": "real",
                "salary": 40_000_000,
                "surplus_value": 46_400_000,
                "contract_label": "surplus_value",
            }
        )
    return pd.DataFrame(rows)


def _sdio_season_frame(year: int, *, pa: float = 500, ip: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "judgeaa01",
                "name_full": "Aaron Judge",
                "name_first": "Aaron",
                "name_last": "Judge",
                "year_id": year,
                "team_id": "NYY",
                "team_name": "Yankees",
                "player_type": "batter",
                "position": "RF",
                "pa": pa,
                "hr": 40,
                "bb": 90,
                "hits": 140,
                "games": 120,
                "ab": 400,
                "so": 130,
                "rbi": 100,
                "sb": 8,
                "woba": None,
                "batting_war": None,
                "ip": ip,
                "fip": None,
                "era": None,
                "whip": None,
                "pitching_so": 0,
                "pitching_bb": 0,
                "pitching_war": None,
                "player_war": None,
                "war_source": "approx",
                "salary": None,
                "surplus_value": None,
                "contract_label": None,
                "stat_source": "sportsdataio",
            }
        ]
    )


def test_window_derivation_matches_as_of_year() -> None:
    assert default_season_window(AS_OF) == [2024, 2025, 2026]
    assert seasons_from_settings({}, AS_OF, environ={}) == [2024, 2025, 2026]


def test_bridge_overlays_sdio_years_missing_from_lahman() -> None:
    lahman = _lahman_frame(2024)
    sdio = pd.concat(
        [_sdio_season_frame(2024), _sdio_season_frame(2025), _sdio_season_frame(2026)],
        ignore_index=True,
    )
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman, sdio, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    years = set(combined["year_id"].astype(int))
    assert years == {2024, 2025, 2026}
    kept = combined.loc[combined["year_id"] == 2024].iloc[0]
    assert kept["player_war"] == pytest.approx(10.8)
    assert kept["war_source"] == "real"
    overlay = combined.loc[combined["year_id"] == 2026].iloc[0]
    assert overlay["stat_source"] == "sportsdataio"
    assert overlay["war_source"] == "approx"
    assert overlay["pa"] == 500
    assert overlay["player_war"] > 0
    assert coverage.active_season == 2026
    assert coverage.active_season_present is True
    assert coverage.active_season_source == "sportsdataio"
    assert coverage.current_season_missing is False
    assert coverage.sdio_in_season is True
    assert coverage.overlay_seasons == [2025, 2026]
    assert (
        decide_current_promote(
            sdio_in_season=coverage.sdio_in_season,
            active_season=coverage.active_season,
            metrics_max_season=max(coverage.seasons_present),
            current_season_missing=coverage.current_season_missing,
        )
        == "promote"
    )


def test_bridge_uses_game_rollup_when_season_stub_is_thin() -> None:
    lahman = _lahman_frame(2024)
    thin = _sdio_season_frame(2026, pa=0, ip=0)
    games = _sdio_season_frame(2026, pa=25, ip=0)
    games["hr"] = 2
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman, thin, games, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    overlay = combined.loc[combined["year_id"] == 2026]
    assert len(overlay) == 1
    assert overlay.iloc[0]["pa"] == 25
    assert overlay.iloc[0]["hr"] == 2
    assert coverage.active_season_present is True
    assert coverage.current_season_missing is False


def test_soft_fail_missing_sdio_does_not_pretend_current_season(tmp_path: Path) -> None:
    lahman = _lahman_frame(2023, 2024)
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman, None, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    assert set(combined["year_id"].astype(int)) == {2023, 2024}
    assert 2026 not in set(combined["year_id"].astype(int))
    assert coverage.current_season_missing is True
    assert coverage.active_season_present is False
    assert coverage.active_season_source is None
    assert coverage.current_season_missing_reason == "sdio_unavailable"
    dest = write_metrics_manifest(tmp_path, coverage)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert dest.name == METRICS_MANIFEST_NAME
    assert payload["current_season_missing"] is True
    assert payload["sdio_in_season"] is False
    assert payload["active_season"] == 2026
    assert payload["season_window"] == [2024, 2025, 2026]
    assert 2026 not in payload["seasons_present"]
    assert (
        decide_current_promote(
            sdio_in_season=coverage.sdio_in_season,
            active_season=coverage.active_season,
            metrics_max_season=max(coverage.seasons_present),
            current_season_missing=coverage.current_season_missing,
        )
        == "skip_soft"
    )


def test_in_season_extract_without_metrics_year_fails_closed_promote() -> None:
    lahman = _lahman_frame(2024)
    report = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": None,
        "current_season_missing": False,
        "endpoints": [
            {"endpoint": "player_season_stats", "ok": True, "season": 2026},
        ],
    }
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman,
        None,
        None,
        as_of_date=AS_OF,
        window=[2024, 2025, 2026],
        extract_report=report,
    )
    assert 2026 not in set(combined["year_id"].astype(int))
    assert coverage.sdio_in_season is True
    assert coverage.current_season_missing is True
    assert coverage.active_season == 2026
    assert max(coverage.seasons_present) < coverage.active_season
    assert (
        decide_current_promote(
            sdio_in_season=coverage.sdio_in_season,
            active_season=coverage.active_season,
            metrics_max_season=max(coverage.seasons_present),
            current_season_missing=coverage.current_season_missing,
        )
        == "fail_closed"
    )


def test_missing_key_extract_skips_promote_without_failing() -> None:
    lahman = _lahman_frame(2023, 2024)
    report = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": "missing_api_key",
        "current_season_missing": True,
        "endpoints": [],
    }
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman,
        None,
        None,
        as_of_date=AS_OF,
        window=[2024, 2025, 2026],
        extract_report=report,
    )
    assert 2026 not in set(combined["year_id"].astype(int))
    assert coverage.sdio_in_season is False
    assert coverage.current_season_missing is True
    assert coverage.current_season_missing_reason == "sdio_unavailable"
    assert (
        decide_current_promote(
            sdio_in_season=coverage.sdio_in_season,
            active_season=coverage.active_season,
            metrics_max_season=max(coverage.seasons_present),
            current_season_missing=coverage.current_season_missing,
        )
        == "skip_soft"
    )


def test_sdio_present_but_empty_active_season_is_flagged() -> None:
    lahman = _lahman_frame(2024)
    sdio = _sdio_season_frame(2025)
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman, sdio, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    assert 2026 not in set(combined["year_id"].astype(int))
    assert coverage.current_season_missing is True
    assert coverage.current_season_missing_reason == "sdio_empty_active_season"
    assert coverage.overlay_seasons == [2025]


def test_cards_pick_latest_including_2026() -> None:
    lahman = _lahman_frame(2024)
    sdio = _sdio_season_frame(2026)
    combined, _coverage = bridge_sdio_player_season_metrics(
        lahman, sdio, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    enriched = enrich_player_season_phase0(combined, as_of_date=AS_OF)
    cards = rank_fantasy_cards(enriched, as_of_date=AS_OF, top_n=1)
    assert cards
    assert {card["season"] for card in cards} == {2026}
    start = next(card for card in cards if card["recommendation_type"] == "start")
    assert start["player"]["name"] == "Aaron Judge"
    assert start["edge"]["war_source"] == "approx"
    assert "vs repl" not in start["share"]["stat_line"]


def test_counting_proxy_is_not_rwar_scale_zero_for_empty_row() -> None:
    empty = pd.Series({"pa": 0, "ip": 0, "hr": 0, "hits": 0, "bb": 0, "sb": 0, "so": 0, "rbi": 0})
    assert approx_vs_replacement_from_counting(empty) == 0.0
    judgeish = pd.Series(
        {"pa": 704, "ip": 0, "hr": 58, "hits": 180, "bb": 133, "sb": 10, "so": 171, "rbi": 144}
    )
    assert approx_vs_replacement_from_counting(judgeish) > 5


def test_warehouse_query_bridges_landed_sdio_frames(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    mapping = {
        ("teams", "teams.json"): "teams.json",
        ("players", "players.json"): "players.json",
        ("player_season_stats", "player_season_stats_2026.json"): "player_season_stats.json",
        ("player_game_stats", f"player_game_stats_{AS_OF}.json"): "player_game_stats.json",
    }
    for (endpoint, filename), fixture in mapping.items():
        write_raw_payload(
            _payload(fixture),
            endpoint=endpoint,
            as_of_date=AS_OF,
            filename=filename,
            raw_dir=raw_dir,
        )
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01"],
            "mlbID": [592450],
            "bbrefID": ["judgeaa01"],
        }
    )
    frames = load_sdio_frames(
        raw_dir,
        as_of_date=AS_OF,
        people=people,
        team_map_path=Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "mlb_team_map.csv",
        run_id="test-run",
    )
    # Fixture season file is labeled 2024 in the payload; rewrite to 2026 for this case.
    season = frames.player_season_stat.copy()
    season["season"] = 2026
    frames.player_season_stat = season

    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.execute(
        "INSERT INTO fact_player_season "
        "(player_id, season_key, team_id, player_type, pa, hr, bb, player_war, war_source) "
        "VALUES ('judgeaa01', 2024, 'NYA', 'batter', 700, 58, 133, 10.8, 'real')"
    )
    loaded = insert_sdio_spine_tables(con, frames)
    assert loaded["player_season_stat"] >= 1
    war = con.execute(
        "SELECT player_war FROM fact_player_season WHERE player_id = 'judgeaa01'"
    ).fetchone()
    assert war == (10.8,)

    sdio_metrics = con.execute(_SDIO_PLAYER_SEASON_QUERY).fetchdf()
    assert not sdio_metrics.empty
    assert int(sdio_metrics.iloc[0]["year_id"]) == 2026
    assert sdio_metrics.iloc[0]["player_id"] == "judgeaa01"

    game_metrics = con.execute(_SDIO_PLAYER_GAME_ROLLUP_QUERY).fetchdf()
    assert not game_metrics.empty

    lahman = _lahman_frame(2024)
    combined, coverage = bridge_sdio_player_season_metrics(
        lahman, sdio_metrics, game_metrics, as_of_date=AS_OF
    )
    assert 2026 in set(combined["year_id"].astype(int))
    assert coverage.current_season_missing is False
    assert combined.loc[combined["year_id"] == 2024, "player_war"].iloc[0] == pytest.approx(10.8)


def _lahman_team_frame(*years: int) -> pd.DataFrame:
    rows = []
    for year in years:
        rows.append(
            {
                "year_id": year,
                "team_name": "New York Yankees",
                "team_id": "NYA",
                "franchise_id": "NYY",
                "league_id": "AL",
                "wins": 94,
                "losses": 68,
                "games": 162,
                "runs_scored": 815,
                "runs_allowed": 668,
                "run_diff": 147,
                "pythag_wins": 95.0,
                "pythag_gap": -1.0,
                "base_runs": 800.0,
                "base_runs_gap": 15.0,
                "team_batting_war": 30.0,
                "team_pitching_war": 15.0,
                "team_total_war": 45.0,
                "war_source": "real",
                "war_win_gap": 1.0,
                "payroll": 279_000_000,
                "max_salary": 40_000_000,
                "median_salary": 8_000_000,
                "top_1_salary_share": 0.14,
                "top_3_salary_share": 0.32,
                "top_5_salary_share": 0.45,
                "gini_salary": 0.55,
                "dead_money_share": 0.08,
                "payroll_per_win": 2_968_085,
                "wins_per_10m": 3.37,
                "run_diff_per_10m": 5.27,
                "cost_per_war": 6_200_000,
                "war_per_1m": 0.16,
                "surplus_value": 81_000_000,
                "window_phase": "contending",
            }
        )
    return pd.DataFrame(rows)


def _sdio_team_season_frame(
    year: int,
    *,
    pa: float = 500,
    ip: float = 0.0,
    team_id: str = "NYA",
    wins: float | None = None,
    games: int = 120,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "year_id": year,
                "team_id": team_id,
                "team_name": "Yankees",
                "franchise_id": "NYY",
                "league_id": "AL",
                "wins": wins,
                "losses": None,
                "games": games,
                "runs_scored": None,
                "runs_allowed": None,
                "pa": pa,
                "hr": 40,
                "bb": 90,
                "hits": 140,
                "ab": 400,
                "so": 130,
                "rbi": 100,
                "sb": 8,
                "ip": ip,
                "pitching_so": 0,
                "pitching_bb": 0,
                "team_batting_war": None,
                "team_pitching_war": None,
                "team_total_war": None,
                "war_source": "approx",
                "payroll": None,
                "stat_source": "sportsdataio",
            }
        ]
    )


def test_team_bridge_overlays_sdio_years_missing_from_lahman() -> None:
    lahman = _lahman_team_frame(2024)
    sdio = pd.concat(
        [
            _sdio_team_season_frame(2024),
            _sdio_team_season_frame(2025),
            _sdio_team_season_frame(2026),
        ],
        ignore_index=True,
    )
    combined, coverage = bridge_sdio_team_season_metrics(
        lahman, sdio, None, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    years = set(combined["year_id"].astype(int))
    assert years == {2024, 2025, 2026}
    assert (combined["year_id"] == 2024).sum() == 1
    kept = combined.loc[combined["year_id"] == 2024].iloc[0]
    assert kept["team_total_war"] == pytest.approx(45.0)
    assert kept["war_source"] == "real"
    assert kept["payroll"] == 279_000_000
    overlay = combined.loc[combined["year_id"] == 2026].iloc[0]
    assert overlay["stat_source"] == "sportsdataio"
    assert overlay["war_source"] == "approx"
    assert overlay["team_id"] == "NYA"
    assert overlay["team_name"] == "New York Yankees"
    assert overlay["team_total_war"] > 0
    assert pd.isna(overlay["payroll"])
    assert coverage.active_season == 2026
    assert coverage.active_season_present is True
    assert coverage.active_season_source == "sportsdataio"
    assert coverage.current_season_missing is False
    assert coverage.overlay_seasons == [2025, 2026]
    assert 2026 in years_from_frame(combined)


def test_team_bridge_uses_game_rollup_when_season_stub_is_thin() -> None:
    lahman = _lahman_team_frame(2024)
    thin = _sdio_team_season_frame(2026, pa=0, ip=0)
    games = _sdio_team_season_frame(2026, pa=25, ip=0)
    games["hr"] = 2
    games["runs_scored"] = 12
    combined, coverage = bridge_sdio_team_season_metrics(
        lahman, thin, games, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    overlay = combined.loc[combined["year_id"] == 2026]
    assert len(overlay) == 1
    assert overlay.iloc[0]["team_total_war"] > 0
    assert overlay.iloc[0]["runs_scored"] == 12
    assert coverage.active_season_present is True
    assert coverage.current_season_missing is False


def test_team_soft_fail_missing_sdio_does_not_pretend_current_season(tmp_path: Path) -> None:
    lahman = _lahman_team_frame(2023, 2024)
    player_combined, player_coverage = bridge_sdio_player_season_metrics(
        _lahman_frame(2023, 2024), None, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    team_combined, team_coverage = bridge_sdio_team_season_metrics(
        lahman, None, None, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    coverage = attach_team_coverage(player_coverage, team_coverage)
    assert set(team_combined["year_id"].astype(int)) == {2023, 2024}
    assert 2026 not in set(team_combined["year_id"].astype(int))
    assert 2026 not in set(player_combined["year_id"].astype(int))
    assert coverage.current_season_missing is True
    assert coverage.team_current_season_missing is True
    assert coverage.team_active_season_present is False
    assert coverage.team_current_season_missing_reason == "sdio_unavailable"
    dest = write_metrics_manifest(tmp_path, coverage)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["current_season_missing"] is True
    assert payload["team_current_season_missing"] is True
    assert payload["active_season"] == 2026
    assert payload["season_window"] == [2024, 2025, 2026]
    assert 2026 not in payload["team_seasons_present"]


def test_current_season_missing_stays_honest_when_team_2026_absent(tmp_path: Path) -> None:
    """Player overlay can have 2026; FO team rails must not look current without it."""
    player_df, player_coverage = bridge_sdio_player_season_metrics(
        _lahman_frame(2024),
        _sdio_season_frame(2026),
        None,
        as_of_date=AS_OF,
        window=[2024, 2025, 2026],
    )
    team_df, team_coverage = bridge_sdio_team_season_metrics(
        _lahman_team_frame(2024),
        None,
        None,
        None,
        as_of_date=AS_OF,
        window=[2024, 2025, 2026],
    )
    assert 2026 in set(player_df["year_id"].astype(int))
    assert player_coverage.current_season_missing is False
    assert 2026 not in set(team_df["year_id"].astype(int))
    coverage = attach_team_coverage(player_coverage, team_coverage)
    assert coverage.current_season_missing is True
    assert coverage.team_current_season_missing is True
    assert coverage.current_season_missing_reason == "sdio_unavailable"
    dest = write_metrics_manifest(tmp_path, coverage)
    payload = json.loads(dest.read_text(encoding="utf-8"))
    assert payload["current_season_missing"] is True
    assert payload["team_current_season_missing"] is True
    assert 2026 not in payload["team_seasons_present"]


def test_team_sdio_present_but_empty_active_season_is_flagged() -> None:
    lahman = _lahman_team_frame(2024)
    sdio = _sdio_team_season_frame(2025)
    combined, coverage = bridge_sdio_team_season_metrics(
        lahman, sdio, None, None, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    assert 2026 not in set(combined["year_id"].astype(int))
    assert coverage.current_season_missing is True
    assert coverage.current_season_missing_reason == "sdio_empty_active_season"
    assert coverage.overlay_seasons == [2025]


def test_team_standings_fill_year_without_counting_overlay() -> None:
    lahman = _lahman_team_frame(2024)
    standings = pd.DataFrame(
        [
            {
                "year_id": 2026,
                "team_id": "NYA",
                "team_name": "New York Yankees",
                "franchise_id": "NYY",
                "league_id": "AL",
                "wins": 55,
                "losses": 40,
                "games": 95,
                "runs_scored": 480,
                "runs_allowed": 410,
                "pa": 0,
                "hr": 0,
                "hits": 0,
                "bb": 0,
                "sb": 0,
                "so": 0,
                "rbi": 0,
                "ip": 0,
                "war_source": "approx",
                "stat_source": "sportsdataio",
            }
        ]
    )
    combined, coverage = bridge_sdio_team_season_metrics(
        lahman, None, None, standings, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    overlay = combined.loc[combined["year_id"] == 2026].iloc[0]
    assert overlay["wins"] == 55
    assert overlay["losses"] == 40
    assert overlay["run_diff"] == 70
    assert overlay["pythag_wins"] > 0
    assert overlay["war_source"] == "approx"
    assert pd.isna(overlay["payroll"])
    assert coverage.current_season_missing is False


def test_one_day_standings_do_not_overwrite_fat_season_rollup() -> None:
    lahman = _lahman_team_frame(2024)
    season = _sdio_team_season_frame(2026, pa=500)
    standings = pd.DataFrame(
        [
            {
                "year_id": 2026,
                "team_id": "NYA",
                "wins": 1,
                "losses": 0,
                "games": 1,
                "runs_scored": 7,
                "runs_allowed": 3,
                "pa": 0,
                "ip": 0,
                "stat_source": "sportsdataio",
            }
        ]
    )
    combined, _coverage = bridge_sdio_team_season_metrics(
        lahman, season, None, standings, as_of_date=AS_OF, window=[2024, 2025, 2026]
    )
    overlay = combined.loc[combined["year_id"] == 2026].iloc[0]
    assert pd.isna(overlay["wins"])
    assert overlay["games"] == 120


def test_warehouse_team_query_bridges_landed_sdio_frames(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    mapping = {
        ("teams", "teams.json"): "teams.json",
        ("players", "players.json"): "players.json",
        ("player_season_stats", "player_season_stats_2026.json"): "player_season_stats.json",
        ("player_game_stats", f"player_game_stats_{AS_OF}.json"): "player_game_stats.json",
        ("games_by_date", f"games_by_date_{AS_OF}.json"): "games_by_date.json",
    }
    for (endpoint, filename), fixture in mapping.items():
        write_raw_payload(
            _payload(fixture),
            endpoint=endpoint,
            as_of_date=AS_OF,
            filename=filename,
            raw_dir=raw_dir,
        )
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01"],
            "mlbID": [592450],
            "bbrefID": ["judgeaa01"],
        }
    )
    frames = load_sdio_frames(
        raw_dir,
        as_of_date=AS_OF,
        people=people,
        team_map_path=Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "mlb_team_map.csv",
        run_id="test-run",
    )
    season = frames.player_season_stat.copy()
    season["season"] = 2026
    frames.player_season_stat = season
    games = frames.player_game_stat.copy()
    games["season"] = 2026
    frames.player_game_stat = games
    game_rows = frames.games.copy()
    game_rows["season"] = 2026
    frames.games = game_rows

    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    loaded = insert_sdio_spine_tables(con, frames)
    assert loaded["player_season_stat"] >= 1
    assert loaded["team"] >= 1

    team_metrics = con.execute(_SDIO_TEAM_SEASON_QUERY).fetchdf()
    assert not team_metrics.empty
    assert int(team_metrics.iloc[0]["year_id"]) == 2026
    assert team_metrics.iloc[0]["team_id"] == "NYA"

    game_metrics = con.execute(_SDIO_TEAM_GAME_ROLLUP_QUERY).fetchdf()
    assert not game_metrics.empty
    standings = con.execute(_SDIO_TEAM_STANDINGS_QUERY).fetchdf()
    assert not standings.empty

    lahman = _lahman_team_frame(2024)
    combined, coverage = bridge_sdio_team_season_metrics(
        lahman, team_metrics, game_metrics, standings, as_of_date=AS_OF
    )
    assert 2026 in set(combined["year_id"].astype(int))
    assert coverage.current_season_missing is False
    yankees_2024 = combined.loc[combined["year_id"] == 2024].iloc[0]
    assert yankees_2024["team_total_war"] == pytest.approx(45.0)
    assert yankees_2024["war_source"] == "real"
    overlay = combined.loc[combined["year_id"] == 2026].iloc[0]
    assert overlay["stat_source"] == "sportsdataio"
    assert overlay["war_source"] == "approx"
    assert pd.isna(overlay["payroll"])
    assert 2026 in years_from_frame(combined)
