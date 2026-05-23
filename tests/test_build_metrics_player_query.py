from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _build_player_metrics_fixture():
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE fact_player_season (
            player_id VARCHAR,
            season_key INTEGER,
            team_id VARCHAR,
            player_type VARCHAR,
            pa INTEGER,
            hr INTEGER,
            bb INTEGER,
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
            team_key VARCHAR,
            team_id VARCHAR,
            franchise_id VARCHAR,
            team_name VARCHAR,
            league_id VARCHAR
        )
        """
    )
    return con


def test_player_query_collapses_traded_player_with_weighted_rates_and_season_team_join() -> None:
    con = _build_player_metrics_fixture()
    try:
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [
                ("traded001", "Traded Star", "Traded", "Star"),
            ],
        )
        con.executemany(
            "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
            [
                ("NYA_2023", "NYA", "NYY", "New York Highlanders", "AL"),
                ("NYA_2024", "NYA", "NYY", "New York Yankees", "AL"),
                ("LAN_2024", "LAN", "LAD", "Los Angeles Dodgers", "NL"),
            ],
        )
        con.executemany(
            """
            INSERT INTO fact_player_season VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "traded001",
                    2024,
                    "NYA",
                    "batter",
                    100,
                    10,
                    20,
                    0.400,
                    2.0,
                    10.0,
                    2.00,
                    3.00,
                    0.5,
                    2.5,
                    1_000_000,
                    8_000_000,
                    "surplus_value",
                ),
                (
                    "traded001",
                    2024,
                    "LAN",
                    "pitcher",
                    10,
                    1,
                    2,
                    0.100,
                    0.1,
                    30.0,
                    5.00,
                    6.00,
                    1.0,
                    1.1,
                    2_000_000,
                    4_000_000,
                    "fair_value",
                ),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()

        assert len(result) == 1
        row = result.iloc[0]
        assert row["player_id"] == "traded001"
        assert row["team_id"] == "NYA"
        assert row["team_name"] == "New York Yankees"
        assert row["pa"] == 110
        assert row["woba"] == pytest.approx(((100 * 0.400) + (10 * 0.100)) / 110)
        assert row["ip"] == pytest.approx(40.0)
        assert row["fip"] == pytest.approx(((10 * 2.00) + (30 * 5.00)) / 40)
        assert row["era"] == pytest.approx(((10 * 3.00) + (30 * 6.00)) / 40)
        assert row["player_war"] == pytest.approx(3.6)
        assert row["salary"] == pytest.approx(3_000_000)
    finally:
        con.close()


def test_player_query_keeps_same_name_players_separate_by_player_id() -> None:
    con = _build_player_metrics_fixture()
    try:
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [
                ("smith001", "Chris Smith", "Chris", "Smith"),
                ("smith002", "Chris Smith", "Chris", "Smith"),
            ],
        )
        con.execute("INSERT INTO dim_team VALUES ('BOS_2024', 'BOS', 'BOS', 'Boston Red Sox', 'AL')")
        con.executemany(
            """
            INSERT INTO fact_player_season VALUES
            (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "smith001",
                    2024,
                    "BOS",
                    "batter",
                    250,
                    12,
                    30,
                    0.330,
                    1.5,
                    None,
                    None,
                    None,
                    None,
                    1.5,
                    750_000,
                    5_000_000,
                    "surplus_value",
                ),
                (
                    "smith002",
                    2024,
                    "BOS",
                    "pitcher",
                    None,
                    None,
                    None,
                    None,
                    None,
                    120.0,
                    3.75,
                    4.10,
                    2.2,
                    2.2,
                    1_250_000,
                    6_000_000,
                    "surplus_value",
                ),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()

        assert set(result["player_id"]) == {"smith001", "smith002"}
        assert result["name_full"].tolist().count("Chris Smith") == 2
        assert result["player_war"].sum() == pytest.approx(3.7)
    finally:
        con.close()
