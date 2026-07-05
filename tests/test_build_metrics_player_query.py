from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def player_metrics_connection() -> duckdb.DuckDBPyConnection:
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
            team_id VARCHAR,
            team_name VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?)",
        [
            ("AAA", "Alpha Aces"),
            ("BBB", "Beta Bears"),
            ("BBB", "Beta Bears"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("traded01", "Traded Player", "Traded", "Player"),
            ("garcia01", "Luis Garcia", "Luis", "Garcia"),
            ("garcia02", "Luis Garcia", "Luis", "Garcia"),
        ],
    )
    yield con
    con.close()


def _insert_player_season(
    con: duckdb.DuckDBPyConnection,
    player_id: str,
    season_key: int,
    team_id: str,
    player_type: str,
    pa: int,
    hr: int,
    bb: int,
    woba: float | None,
    batting_war: float,
    ip: float,
    fip: float | None,
    era: float | None,
    pitching_war: float,
    player_war: float,
    salary: float,
    surplus_value: float,
    contract_label: str,
) -> None:
    con.execute(
        """
        INSERT INTO fact_player_season VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            player_id,
            season_key,
            team_id,
            player_type,
            pa,
            hr,
            bb,
            woba,
            batting_war,
            ip,
            fip,
            era,
            pitching_war,
            player_war,
            salary,
            surplus_value,
            contract_label,
        ],
    )


def test_player_query_collapses_traded_stints_to_one_player_season(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    _insert_player_season(
        con,
        "traded01",
        2024,
        "AAA",
        "batter",
        200,
        10,
        20,
        0.320,
        1.5,
        0,
        None,
        None,
        0,
        1.5,
        3_000_000,
        9_000_000,
        "fair_value",
    )
    _insert_player_season(
        con,
        "traded01",
        2024,
        "BBB",
        "batter",
        300,
        15,
        30,
        0.360,
        3.0,
        0,
        None,
        None,
        0,
        3.0,
        5_000_000,
        19_000_000,
        "surplus_value",
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded01"
    assert row["year_id"] == 2024
    assert row["team_name"] == "Beta Bears"
    assert row["pa"] == 500
    assert row["hr"] == 25
    assert row["bb"] == 50
    assert row["batting_war"] == pytest.approx(4.5)
    assert row["player_war"] == pytest.approx(4.5)
    assert row["salary"] == pytest.approx(8_000_000)
    assert row["surplus_value"] == pytest.approx(28_000_000)
    assert row["contract_label"] == "surplus_value"


def test_player_query_keeps_same_name_players_distinct(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    _insert_player_season(
        con,
        "garcia01",
        2024,
        "AAA",
        "pitcher",
        0,
        0,
        0,
        None,
        0,
        120.0,
        3.45,
        3.70,
        2.1,
        2.1,
        2_000_000,
        14_800_000,
        "surplus_value",
    )
    _insert_player_season(
        con,
        "garcia02",
        2024,
        "BBB",
        "batter",
        450,
        18,
        44,
        0.335,
        2.4,
        0,
        None,
        None,
        0,
        2.4,
        6_000_000,
        13_200_000,
        "fair_value",
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 2
    assert set(result["name_full"]) == {"Luis Garcia"}
    assert set(result["player_id"]) == {"garcia01", "garcia02"}
