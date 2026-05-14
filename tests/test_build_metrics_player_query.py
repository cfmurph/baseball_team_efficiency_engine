from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


def _build_player_metrics_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany(
        """
        INSERT INTO dim_team (team_key, team_id, franchise_id, team_name, league_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("AAA_2023", "AAA", "FRA", "Old Alpha Name", "AL"),
            ("AAA_2024", "AAA", "FRA", "Alpha Club", "AL"),
            ("BBB_2024", "BBB", "FRB", "Beta Club", "NL"),
        ],
    )
    con.executemany(
        """
        INSERT INTO dim_player (
            player_id, name_first, name_last, name_full, birth_year, birth_country, throws, bats
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("traded01", "Casey", "Traveler", "Casey Traveler", 1990, "USA", "R", "L"),
            ("pitcher01", "Morgan", "Starter", "Morgan Starter", 1988, "USA", "L", "R"),
            ("shared01", "Alex", "Smith", "Alex Smith", 1991, "USA", "R", "R"),
            ("shared02", "Alex", "Smith", "Alex Smith", 1994, "USA", "L", "L"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season (
            player_id, season_key, team_id, player_type,
            pa, hr, bb, woba, batting_war,
            ip, fip, era, pitching_war,
            player_war, salary, surplus_value, contract_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "traded01", 2024, "AAA", "batter",
                100, 10, 20, 0.300, 1.0,
                0, None, None, 0,
                1.0, 1_000_000, 7_000_000, "surplus_value",
            ),
            (
                "traded01", 2024, "BBB", "both",
                300, 20, 40, 0.400, 2.0,
                0, None, None, 0,
                2.0, 2_000_000, 14_000_000, "star",
            ),
            (
                "pitcher01", 2024, "AAA", "pitcher",
                0, 0, 0, None, 0,
                10, 5.00, 6.00, 0.2,
                0.2, 500_000, 1_000_000, "overpaid",
            ),
            (
                "pitcher01", 2024, "BBB", "pitcher",
                0, 0, 0, None, 0,
                30, 3.00, 2.00, 0.8,
                0.8, 700_000, 3_000_000, "surplus_value",
            ),
            (
                "shared01", 2024, "AAA", "batter",
                50, 5, 5, 0.350, 0.5,
                0, None, None, 0,
                0.5, 600_000, 2_000_000, "surplus_value",
            ),
            (
                "shared02", 2024, "AAA", "batter",
                75, 7, 8, 0.360, 0.7,
                0, None, None, 0,
                0.7, 800_000, 2_500_000, "surplus_value",
            ),
        ],
    )
    return con


def test_player_query_consolidates_traded_players_without_team_history_fanout() -> None:
    con = _build_player_metrics_fixture()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    traded = result[result["player_id"] == "traded01"]
    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Club"
    assert row["player_type"] == "both"
    assert row["pa"] == pytest.approx(400)
    assert row["hr"] == pytest.approx(30)
    assert row["bb"] == pytest.approx(60)
    assert row["woba"] == pytest.approx(0.375)
    assert row["batting_war"] == pytest.approx(3.0)
    assert row["player_war"] == pytest.approx(3.0)
    assert row["salary"] == pytest.approx(3_000_000)
    assert row["surplus_value"] == pytest.approx(21_000_000)
    assert row["contract_label"] == "star"
    assert "Old Alpha Name" not in set(result["team_name"])


def test_player_query_weights_pitching_rates_by_innings_pitched() -> None:
    con = _build_player_metrics_fixture()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    row = result[result["player_id"] == "pitcher01"].iloc[0]
    assert row["ip"] == pytest.approx(40)
    assert row["fip"] == pytest.approx(3.5)
    assert row["era"] == pytest.approx(3.0)
    assert row["pitching_war"] == pytest.approx(1.0)
    assert row["team_name"] == "Beta Club"


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    con = _build_player_metrics_fixture()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    same_name = result[result["name_full"] == "Alex Smith"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["shared01", "shared02"]
    assert same_name["pa"].tolist() == [50, 75]
