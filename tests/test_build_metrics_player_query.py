from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


def _player_metrics_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany("INSERT INTO dim_season (season_key, year_id) VALUES (?, ?)", [(2024, 2024)])
    con.executemany(
        """
        INSERT INTO dim_team (team_key, team_id, franchise_id, team_name, league_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("NYY_2023", "NYY", "NYY", "New York Yankees", "AL"),
            ("NYY_2024", "NYY", "NYY", "New York Yankees", "AL"),
            ("LAD_2024", "LAD", "LAD", "Los Angeles Dodgers", "NL"),
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
            ("trade01", "Trade", "Target", "Trade Target", 1990, "USA", "R", "R"),
            ("youngc01", "Chris", "Young", "Chris Young", 1983, "USA", "R", "R"),
            ("youngc02", "Chris", "Young", "Chris Young", 1993, "USA", "R", "L"),
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
                "trade01",
                2024,
                "NYY",
                "batter",
                100,
                5,
                10,
                0.300,
                1.0,
                None,
                None,
                None,
                0,
                1.0,
                1_000_000,
                7_000_000,
                "surplus_value",
            ),
            (
                "trade01",
                2024,
                "LAD",
                "pitcher",
                20,
                1,
                2,
                0.400,
                0.5,
                40,
                3.50,
                2.80,
                2.5,
                3.0,
                2_000_000,
                22_000_000,
                "fair_value",
            ),
            (
                "youngc01",
                2024,
                "NYY",
                "batter",
                10,
                1,
                1,
                0.250,
                0.1,
                None,
                None,
                None,
                0,
                0.1,
                500_000,
                300_000,
                "fair_value",
            ),
            (
                "youngc02",
                2024,
                "LAD",
                "pitcher",
                None,
                None,
                None,
                None,
                0,
                25,
                4.10,
                3.90,
                0.8,
                0.8,
                750_000,
                5_650_000,
                "surplus_value",
            ),
        ],
    )
    return con


def test_player_query_collapses_traded_stints_to_one_player_season() -> None:
    con = _player_metrics_connection()
    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    traded = df[df["player_id"] == "trade01"]

    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "LAD"
    assert row["team_name"] == "Los Angeles Dodgers"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 120
    assert row["hr"] == 6
    assert row["bb"] == 12
    assert row["ip"] == 40
    assert row["batting_war"] == pytest.approx(1.5)
    assert row["pitching_war"] == pytest.approx(2.5)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 29_000_000
    assert row["contract_label"] == "fair_value"


def test_player_query_does_not_fan_out_repeated_team_ids_or_merge_same_names() -> None:
    con = _player_metrics_connection()
    try:
        df = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(df) == 3
    assert not df.duplicated(["player_id", "year_id"]).any()

    nyy_player = df[df["player_id"] == "youngc01"].iloc[0]
    assert nyy_player["pa"] == 10
    assert nyy_player["salary"] == 500_000

    same_name_players = df[df["name_full"] == "Chris Young"]
    assert set(same_name_players["player_id"]) == {"youngc01", "youngc02"}
    assert len(same_name_players) == 2
