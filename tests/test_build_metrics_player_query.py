from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _build_player_query_fixture() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE dim_team (
            team_key VARCHAR,
            team_id VARCHAR,
            franchise_id VARCHAR,
            team_name VARCHAR,
            league_id VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dim_player (
            player_id VARCHAR,
            name_first VARCHAR,
            name_last VARCHAR,
            name_full VARCHAR,
            birth_year INTEGER,
            birth_country VARCHAR,
            throws VARCHAR,
            bats VARCHAR
        )
        """
    )
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
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("AAA_2020", "AAA", "AAA", "Old City Anchors", "AL"),
            ("AAA_2021", "AAA", "AAA", "New City Anchors", "AL"),
            ("BBB_2020", "BBB", "BBB", "Bay Bears", "NL"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("traded01", "Trade", "Case", "Trade Case", 1990, "USA", "R", "L"),
            ("alex01", "Alex", "Gonzalez", "Alex Gonzalez", 1977, "USA", "R", "R"),
            ("alex02", "Alex", "Gonzalez", "Alex Gonzalez", 1981, "USA", "R", "R"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "traded01",
                2020,
                "AAA",
                "batter",
                100.0,
                8.0,
                12.0,
                0.300,
                0.8,
                10.0,
                3.00,
                4.00,
                0.2,
                1.0,
                1_000_000.0,
                2_000_000.0,
                "pre_arb",
            ),
            (
                "traded01",
                2020,
                "BBB",
                "both",
                300.0,
                20.0,
                48.0,
                0.400,
                2.0,
                30.0,
                5.00,
                2.00,
                1.0,
                3.0,
                2_000_000.0,
                3_000_000.0,
                "free_agent",
            ),
            (
                "alex01",
                2020,
                "AAA",
                "batter",
                50.0,
                1.0,
                5.0,
                0.250,
                0.1,
                0.0,
                None,
                None,
                0.0,
                0.1,
                500_000.0,
                750_000.0,
                "pre_arb",
            ),
            (
                "alex02",
                2020,
                "BBB",
                "batter",
                75.0,
                2.0,
                7.0,
                0.275,
                0.2,
                0.0,
                None,
                None,
                0.0,
                0.2,
                600_000.0,
                900_000.0,
                "pre_arb",
            ),
        ],
    )
    return con


def test_player_query_rolls_up_traded_player_with_weighted_rates() -> None:
    con = _build_player_query_fixture()

    df = con.execute(_PLAYER_QUERY).fetchdf()

    traded = df[df["player_id"] == "traded01"]
    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Bay Bears"
    assert row["player_type"] == "both"
    assert row["pa"] == pytest.approx(400.0)
    assert row["hr"] == pytest.approx(28.0)
    assert row["bb"] == pytest.approx(60.0)
    assert row["ip"] == pytest.approx(40.0)
    assert row["woba"] == pytest.approx(0.375)
    assert row["fip"] == pytest.approx(4.5)
    assert row["era"] == pytest.approx(2.5)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000.0)
    assert row["surplus_value"] == pytest.approx(5_000_000.0)
    assert row["contract_label"] == "free_agent"

    con.close()


def test_player_query_uses_season_specific_team_dimension_without_fanout() -> None:
    con = _build_player_query_fixture()

    df = con.execute(_PLAYER_QUERY).fetchdf()

    alex_one = df[df["player_id"] == "alex01"].iloc[0]
    assert alex_one["team_name"] == "Old City Anchors"
    assert alex_one["pa"] == pytest.approx(50.0)
    assert alex_one["salary"] == pytest.approx(500_000.0)

    con.close()


def test_player_query_preserves_distinct_same_name_players() -> None:
    con = _build_player_query_fixture()

    df = con.execute(_PLAYER_QUERY).fetchdf()

    same_name = df[df["name_full"] == "Alex Gonzalez"]
    assert set(same_name["player_id"]) == {"alex01", "alex02"}
    assert len(same_name) == 2

    con.close()
