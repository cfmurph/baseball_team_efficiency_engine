from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _fixture_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE fact_player_season (
            player_id VARCHAR,
            season_key INTEGER,
            team_id VARCHAR,
            player_type VARCHAR,
            pa DOUBLE,
            hr DOUBLE,
            bb DOUBLE,
            woba DOUBLE,
            batting_war DOUBLE,
            ip DOUBLE,
            fip DOUBLE,
            era DOUBLE,
            pitching_war DOUBLE,
            player_war DOUBLE,
            salary DOUBLE,
            surplus_value DOUBLE,
            contract_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dim_player (
            player_id VARCHAR,
            name_full VARCHAR,
            name_first VARCHAR,
            name_last VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dim_team (
            team_id VARCHAR,
            team_name VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?)",
        [
            ("AAA", "Alpha Club"),
            ("AAA", "Alpha Renamed"),
            ("BBB", "Beta Club"),
            ("BBB", "Beta Renamed"),
            ("CCC", "Gamma Club"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("traded01", "Traded Player", "Traded", "Player"),
            ("youngch01", "Chris Young", "Chris", "Young"),
            ("youngch02", "Chris Young", "Chris", "Young"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "traded01",
                2024,
                "AAA",
                "both",
                100,
                10,
                20,
                0.300,
                1.0,
                100,
                5.0,
                6.0,
                0.5,
                1.0,
                1_000_000,
                7_000_000,
                "surplus_value",
            ),
            (
                "traded01",
                2024,
                "BBB",
                "pitcher",
                300,
                20,
                60,
                0.400,
                2.0,
                50,
                3.0,
                4.0,
                1.5,
                3.0,
                2_000_000,
                22_000_000,
                "fair_value",
            ),
            (
                "youngch01",
                2024,
                "CCC",
                "batter",
                200,
                8,
                30,
                0.320,
                1.2,
                0,
                None,
                None,
                0,
                1.2,
                750_000,
                8_850_000,
                "surplus_value",
            ),
            (
                "youngch02",
                2024,
                "CCC",
                "pitcher",
                0,
                0,
                0,
                None,
                0,
                80,
                4.0,
                4.5,
                1.8,
                1.8,
                1_500_000,
                12_900_000,
                "surplus_value",
            ),
        ],
    )
    return con


def test_player_query_aggregates_traded_player_without_team_dimension_fanout() -> None:
    con = _fixture_connection()

    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    traded = df[df["player_id"] == "traded01"]
    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["year_id"] == 2024
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Club"
    assert row["player_type"] == "both"
    assert row["pa"] == pytest.approx(400)
    assert row["hr"] == pytest.approx(30)
    assert row["bb"] == pytest.approx(80)
    assert row["ip"] == pytest.approx(150)
    assert row["batting_war"] == pytest.approx(3.0)
    assert row["pitching_war"] == pytest.approx(2.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000)
    assert row["surplus_value"] == pytest.approx(29_000_000)
    assert row["contract_label"] == "fair_value"


def test_player_query_weights_rate_stats_by_playing_time() -> None:
    con = _fixture_connection()

    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    row = df[df["player_id"] == "traded01"].iloc[0]
    assert row["woba"] == pytest.approx(((0.300 * 100) + (0.400 * 300)) / 400)
    assert row["fip"] == pytest.approx(((5.0 * 100) + (3.0 * 50)) / 150)
    assert row["era"] == pytest.approx(((6.0 * 100) + (4.0 * 50)) / 150)


def test_player_query_keeps_same_name_players_distinct_by_player_id() -> None:
    con = _fixture_connection()

    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    same_name = df[df["name_full"] == "Chris Young"]
    assert len(same_name) == 2
    assert set(same_name["player_id"]) == {"youngch01", "youngch02"}
