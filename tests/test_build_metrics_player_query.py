from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _run_player_query(
    fact_player_season: pd.DataFrame,
    dim_player: pd.DataFrame,
    dim_team: pd.DataFrame,
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.register("fact_player_season", fact_player_season)
    con.register("dim_player", dim_player)
    con.register("dim_team", dim_team)
    try:
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def test_player_query_consolidates_traded_player_with_weighted_rates() -> None:
    result = _run_player_query(
        fact_player_season=pd.DataFrame(
            [
                {
                    "player_id": "traded1",
                    "season_key": 2024,
                    "team_id": "AAA",
                    "player_type": "batter",
                    "pa": 100,
                    "hr": 5,
                    "bb": 10,
                    "woba": 0.500,
                    "batting_war": 1.0,
                    "ip": 10.0,
                    "fip": 2.00,
                    "era": 3.00,
                    "pitching_war": 0.5,
                    "player_war": 1.5,
                    "salary": 1_000_000,
                    "surplus_value": 10_000_000,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "traded1",
                    "season_key": 2024,
                    "team_id": "BBB",
                    "player_type": "both",
                    "pa": 300,
                    "hr": 15,
                    "bb": 30,
                    "woba": 0.300,
                    "batting_war": 2.0,
                    "ip": 30.0,
                    "fip": 4.00,
                    "era": 5.00,
                    "pitching_war": 0.5,
                    "player_war": 2.5,
                    "salary": 2_000_000,
                    "surplus_value": 20_000_000,
                    "contract_label": "overpaid",
                },
            ]
        ),
        dim_player=pd.DataFrame(
            [
                {
                    "player_id": "traded1",
                    "name_full": "Traded Player",
                    "name_first": "Traded",
                    "name_last": "Player",
                }
            ]
        ),
        dim_team=pd.DataFrame(
            [
                {"team_key": "AAA_2024", "team_id": "AAA", "team_name": "Alpha"},
                {"team_key": "BBB_2023", "team_id": "BBB", "team_name": "Old Beta"},
                {"team_key": "BBB_2024", "team_id": "BBB", "team_name": "Beta"},
            ]
        ),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded1"
    assert row["year_id"] == 2024
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta"
    assert row["player_type"] == "both"
    assert row["pa"] == 400
    assert row["hr"] == 20
    assert row["bb"] == 40
    assert row["woba"] == pytest.approx((0.500 * 100 + 0.300 * 300) / 400)
    assert row["batting_war"] == pytest.approx(3.0)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx((2.00 * 10 + 4.00 * 30) / 40)
    assert row["era"] == pytest.approx((3.00 * 10 + 5.00 * 30) / 40)
    assert row["pitching_war"] == pytest.approx(1.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 30_000_000
    assert row["contract_label"] == "overpaid"


def test_player_query_preserves_same_name_players_as_distinct_people() -> None:
    result = _run_player_query(
        fact_player_season=pd.DataFrame(
            [
                {
                    "player_id": "smith-a",
                    "season_key": 2024,
                    "team_id": "AAA",
                    "player_type": "batter",
                    "pa": 250,
                    "hr": 8,
                    "bb": 20,
                    "woba": 0.320,
                    "batting_war": 1.2,
                    "ip": 0.0,
                    "fip": None,
                    "era": None,
                    "pitching_war": 0.0,
                    "player_war": 1.2,
                    "salary": 750_000,
                    "surplus_value": 8_000_000,
                    "contract_label": "surplus_value",
                },
                {
                    "player_id": "smith-b",
                    "season_key": 2024,
                    "team_id": "BBB",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 120.0,
                    "fip": 3.50,
                    "era": 3.80,
                    "pitching_war": 2.4,
                    "player_war": 2.4,
                    "salary": 1_500_000,
                    "surplus_value": 17_000_000,
                    "contract_label": "surplus_value",
                },
            ]
        ),
        dim_player=pd.DataFrame(
            [
                {
                    "player_id": "smith-a",
                    "name_full": "Alex Smith",
                    "name_first": "Alex",
                    "name_last": "Smith",
                },
                {
                    "player_id": "smith-b",
                    "name_full": "Alex Smith",
                    "name_first": "Alex",
                    "name_last": "Smith",
                },
            ]
        ),
        dim_team=pd.DataFrame(
            [
                {"team_key": "AAA_2024", "team_id": "AAA", "team_name": "Alpha"},
                {"team_key": "BBB_2024", "team_id": "BBB", "team_name": "Beta"},
            ]
        ),
    )

    assert len(result) == 2
    assert set(result["player_id"]) == {"smith-a", "smith-b"}
    assert result["name_full"].tolist() == ["Alex Smith", "Alex Smith"]
