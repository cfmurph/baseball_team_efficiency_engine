from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


def _connect_with_player_fixtures() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("NYA_2022", "NYA", "NYY", "New York Highlanders", "AL"),
            ("NYA_2023", "NYA", "NYY", "New York Yankees", "AL"),
            ("NYA_2024", "NYA", "NYY", "New York Pinstripes", "AL"),
            ("BOS_2023", "BOS", "BRS", "Boston Red Sox", "AL"),
            ("LAN_2023", "LAN", "LAD", "Los Angeles Dodgers", "NL"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("traded01", "Taylor", "Traded", "Taylor Traded", 1990, "USA", "R", "L"),
            ("youngch01", "Chris", "Young", "Chris Young", 1983, "USA", "R", "R"),
            ("youngch02", "Chris", "Young", "Chris Young", 1979, "USA", "R", "R"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "traded01",
                2023,
                "NYA",
                "batter",
                100.0,
                10.0,
                20.0,
                0.500,
                1.0,
                10.0,
                5.00,
                4.00,
                0.5,
                1.5,
                1_000_000.0,
                11_000_000.0,
                "surplus_value",
            ),
            (
                "traded01",
                2023,
                "BOS",
                "pitcher",
                300.0,
                20.0,
                30.0,
                0.300,
                2.0,
                30.0,
                3.00,
                2.00,
                1.5,
                3.5,
                4_000_000.0,
                24_000_000.0,
                "fair_value",
            ),
            (
                "youngch01",
                2023,
                "NYA",
                "batter",
                200.0,
                15.0,
                25.0,
                0.360,
                2.5,
                0.0,
                None,
                None,
                0.0,
                2.5,
                750_000.0,
                19_250_000.0,
                "surplus_value",
            ),
            (
                "youngch02",
                2023,
                "LAN",
                "pitcher",
                0.0,
                0.0,
                0.0,
                None,
                0.0,
                150.0,
                3.80,
                3.50,
                3.0,
                3.0,
                8_000_000.0,
                16_000_000.0,
                "fair_value",
            ),
        ],
    )
    return con


def test_player_query_rolls_traded_stints_into_weighted_player_season() -> None:
    con = _connect_with_player_fixtures()
    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    row = df[df["player_id"] == "traded01"].iloc[0]
    assert row["year_id"] == 2023
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["contract_label"] == "fair_value"
    assert row["player_type"] == "pitcher"

    assert row["pa"] == pytest.approx(400.0)
    assert row["hr"] == pytest.approx(30.0)
    assert row["bb"] == pytest.approx(50.0)
    assert row["woba"] == pytest.approx(((100.0 * 0.500) + (300.0 * 0.300)) / 400.0)
    assert row["batting_war"] == pytest.approx(3.0)

    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(((10.0 * 5.00) + (30.0 * 3.00)) / 40.0)
    assert row["era"] == pytest.approx(((10.0 * 4.00) + (30.0 * 2.00)) / 40.0)
    assert row["pitching_war"] == pytest.approx(2.0)
    assert row["player_war"] == pytest.approx(5.0)
    assert row["salary"] == pytest.approx(5_000_000.0)
    assert row["surplus_value"] == pytest.approx(35_000_000.0)


def test_player_query_uses_season_specific_team_join_without_name_fanout() -> None:
    con = _connect_with_player_fixtures()
    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    row = df[df["player_id"] == "youngch01"].iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["pa"] == pytest.approx(200.0)
    assert row["hr"] == pytest.approx(15.0)
    assert row["player_war"] == pytest.approx(2.5)


def test_player_query_preserves_distinct_same_name_players() -> None:
    con = _connect_with_player_fixtures()
    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    same_name = df[df["name_full"] == "Chris Young"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["youngch01", "youngch02"]
    assert same_name["team_name"].tolist() == ["New York Yankees", "Los Angeles Dodgers"]
