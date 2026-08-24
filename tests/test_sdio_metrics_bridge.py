"""SDIO → player_season_metrics overlay and current-season coverage (#131)."""
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
    approx_vs_replacement_from_counting,
    bridge_sdio_player_season_metrics,
    enrich_player_season_phase0,
    write_metrics_manifest,
)
from src.baseball_analytics.fantasy import rank_fantasy_cards
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.sportsdataio import (
    default_season_window,
    load_sdio_frames,
    seasons_from_settings,
    write_raw_payload,
)
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
    assert coverage.overlay_seasons == [2025, 2026]


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
    assert payload["active_season"] == 2026
    assert payload["season_window"] == [2024, 2025, 2026]
    assert 2026 not in payload["seasons_present"]


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
