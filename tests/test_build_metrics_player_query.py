from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _player_query_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
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
            team_name VARCHAR
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
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("hitter01", "Shared Name", "Shared", "Name"),
            ("hitter02", "Shared Name", "Shared", "Name"),
            ("pitcher01", "Pitcher One", "Pitcher", "One"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?)",
        [
            ("AAA_2020", "AAA", "Alpha 2020"),
            ("AAA_2021", "AAA", "Alpha 2021"),
            ("BBB_2020", "BBB", "Beta 2020"),
            ("BBB_2021", "BBB", "Beta 2021"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("hitter01", 2020, "AAA", "batter", 100, 10, 20, 0.400, 2.0, None, None, None, 0.0, 2.0, 1_000_000, 7_000_000, "surplus"),
            ("hitter01", 2020, "BBB", "batter", 300, 20, 40, 0.300, 5.0, None, None, None, 0.0, 5.0, 2_000_000, 18_000_000, "star_value"),
            ("hitter02", 2020, "AAA", "batter", 50, 1, 5, 0.250, 0.5, None, None, None, 0.0, 0.5, 600_000, 1_400_000, "fair_value"),
            ("pitcher01", 2020, "AAA", "pitcher", 0, 0, 0, None, 0.0, 50, 3.00, 4.00, 1.0, 1.0, 1_500_000, 2_500_000, "fair_value"),
            ("pitcher01", 2020, "BBB", "pitcher", 0, 0, 0, None, 0.0, 150, 4.00, 2.00, 3.0, 3.0, 3_500_000, 8_500_000, "surplus"),
        ],
    )
    return con


def test_player_query_keeps_one_row_per_player_season_and_season_specific_team_names() -> None:
    con = _player_query_connection()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert not result.duplicated(["player_id", "year_id"]).any()
    assert set(result["player_id"]) == {"hitter01", "hitter02", "pitcher01"}

    hitter = result[result["player_id"] == "hitter01"].iloc[0]
    assert hitter["team_id"] == "BBB"
    assert hitter["team_name"] == "Beta 2020"
    assert hitter["pa"] == 400
    assert hitter["player_war"] == pytest.approx(7.0)
    assert hitter["salary"] == 3_000_000
    assert hitter["surplus_value"] == 25_000_000
    assert hitter["contract_label"] == "star_value"


def test_player_query_weights_rate_stats_by_opportunity() -> None:
    con = _player_query_connection()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    hitter = result[result["player_id"] == "hitter01"].iloc[0]
    pitcher = result[result["player_id"] == "pitcher01"].iloc[0]

    assert hitter["woba"] == pytest.approx(((0.400 * 100) + (0.300 * 300)) / 400)
    assert pitcher["fip"] == pytest.approx(((3.00 * 50) + (4.00 * 150)) / 200)
    assert pitcher["era"] == pytest.approx(((4.00 * 50) + (2.00 * 150)) / 200)


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    con = _player_query_connection()
    try:
        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    same_name = result[result["name_full"] == "Shared Name"]
    assert same_name["player_id"].tolist() == ["hitter01", "hitter02"]
