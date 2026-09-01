from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform import build_metrics
from pipeline.transform.build_metrics import (
    PHASE0_PLAYER_FIELDS,
    attach_published_individual_lines,
    enrich_player_season_phase0,
)

pytestmark = pytest.mark.integration


def test_efficiency_labels_bucket_wins_per_10m() -> None:
    df = pd.DataFrame({"wins_per_10m": [0.3, 0.8, 1.2, 2.0]})

    result = build_metrics._efficiency_labels(df)

    assert result["efficiency_label"].astype(str).tolist() == [
        "low",
        "below_avg",
        "above_avg",
        "elite",
    ]


def test_contract_exports_filter_and_sort_material_contract_risks() -> None:
    player_df = pd.DataFrame(
        {
            "name_full": ["Value Bat", "Replacement", "Injured Ace", "Bench Deal"],
            "player_war": [4.0, 0.0, -0.5, 1.0],
            "salary": [2_000_000, 1_000_000, 25_000_000, 500_000],
            "surplus_value": [28_000_000, 5_000_000, -24_000_000, -2_000_000],
            "contract_label": ["surplus_value", "fair_value", "dead_money", "overpaid"],
        }
    )

    top_value = build_metrics._top_value_players(player_df, n=3)
    worst = build_metrics._worst_contracts(player_df, n=3)
    dead_money = build_metrics._dead_money_leaders(player_df)

    assert top_value["name_full"].tolist() == ["Value Bat", "Bench Deal"]
    assert worst["name_full"].tolist() == ["Injured Ace", "Replacement", "Value Bat"]
    assert dead_money["name_full"].tolist() == ["Injured Ace"]


def test_window_summary_keeps_latest_phase_per_team() -> None:
    team_df = pd.DataFrame(
        {
            "team_name": ["Aces", "Bears", "Aces", "Bears"],
            "year_id": [2022, 2021, 2024, 2023],
            "window_phase": ["building", "retooling", "contending", "rebuilding"],
            "wins": [78, 85, 96, 67],
            "payroll": [90_000_000, 120_000_000, 150_000_000, 80_000_000],
            "team_total_war": [30.0, 35.0, 45.0, 22.0],
        }
    )

    result = build_metrics._window_summary(team_df).sort_values("team_name").reset_index(drop=True)

    assert result[["team_name", "year_id", "window_phase", "wins"]].to_dict("records") == [
        {"team_name": "Aces", "year_id": 2024, "window_phase": "contending", "wins": 96},
        {"team_name": "Bears", "year_id": 2023, "window_phase": "rebuilding", "wins": 67},
    ]


def test_enrich_player_season_phase0_adds_aliases_ranks_and_keeps_grain() -> None:
    player_df = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "name_full": ["Ace", "Bat", "Util"],
            "team_name": ["Aces", "Bears", "Aces"],
            "year_id": [2015, 2015, 2015],
            "player_type": ["pitcher", "batter", "batter"],
            "player_war": [6.0, 4.0, 4.0],
            "war_source": ["real", "approx", "real"],
            "salary": [12_000_000, 2_000_000, 8_000_000],
            "surplus_value": [36_000_000, 30_000_000, 24_000_000],
        }
    )

    result = enrich_player_season_phase0(player_df, as_of_date="2026-08-23")

    for field in PHASE0_PLAYER_FIELDS:
        assert field in result.columns, field
    assert list(result["player_name"]) == ["Ace", "Bat", "Util"]
    assert list(result["team"]) == ["Aces", "Bears", "Aces"]
    assert list(result["season"]) == [2015, 2015, 2015]
    assert list(result["position"]) == ["P", "UTIL", "UTIL"]
    assert list(result["war"]) == [6.0, 4.0, 4.0]
    assert list(result["vs_replacement"]) == [6.0, 4.0, 4.0]
    assert list(result["edge"]) == [36_000_000, 30_000_000, 24_000_000]
    assert result["as_of_date"].unique().tolist() == ["2026-08-23"]
    assert result["rank_overall"].tolist() == [1, 2, 2]
    assert result.loc[result["player_id"] == "a", "rank_at_position"].iloc[0] == 1
    assert result.duplicated(["player_id", "season"]).sum() == 0
    assert result["cost_per_war"].iloc[0] == pytest.approx(2_000_000)
    # Additive — dashboard columns remain.
    assert "name_full" in result.columns and "team_name" in result.columns


def test_table_has_rows_handles_present_empty_and_missing_tables() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE present_empty (id INTEGER)")
        con.execute("CREATE TABLE present_with_rows (id INTEGER)")
        con.execute("INSERT INTO present_with_rows VALUES (1)")

        assert build_metrics._table_has_rows(con, "present_empty") is False
        assert build_metrics._table_has_rows(con, "present_with_rows") is True
        assert build_metrics._table_has_rows(con, "missing_table") is False
    finally:
        con.close()


def test_attach_published_lines_fills_lahman_counting_and_fielding() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "solerjo01"],
            "year_id": [2025, 2026],
            "season": [2025, 2026],
            "pa": [680, 210],
            "player_war": [10.8, 0.2],
        }
    )
    batting = pd.DataFrame(
        {
            "playerID": ["judgeaa01"],
            "yearID": [2025],
            "G": [158],
            "AB": [580],
            "R": [122],
            "H": [180],
            "X2B": [36],
            "X3B": [1],
            "HR": [58],
            "RBI": [144],
            "SB": [10],
            "BB": [130],
            "SO": [170],
        }
    )
    fielding = pd.DataFrame(
        {
            "playerID": ["judgeaa01"],
            "yearID": [2025],
            "POS": ["RF"],
            "G": [150],
            "GS": [148],
            "InnOuts": [3915],
            "PO": [361],
            "A": [8],
            "E": [4],
            "DP": [1],
        }
    )
    out = attach_published_individual_lines(players, batting=batting, fielding=fielding)
    judge = out.loc[out["player_id"] == "judgeaa01"].iloc[0]
    assert judge["runs"] == 122
    assert judge["doubles"] == 36
    assert judge["putouts"] == 361
    assert judge["fpct"] == pytest.approx(0.989)
    assert "RF" in str(judge["fielding_json"])
    soler = out.loc[out["player_id"] == "solerjo01"].iloc[0]
    assert pd.isna(soler.get("putouts")) or soler.get("putouts") in (None, "")
    assert soler.get("fielding_json") in (None, "") or pd.isna(soler.get("fielding_json"))
