from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(
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
    connection.execute(
        """
        CREATE TABLE dim_player (
            player_id VARCHAR,
            name_full VARCHAR,
            name_first VARCHAR,
            name_last VARCHAR
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE dim_team (
            team_key VARCHAR,
            team_id VARCHAR,
            team_name VARCHAR
        )
        """
    )
    try:
        yield connection
    finally:
        connection.close()


def test_player_query_aggregates_traded_player_without_historical_team_fanout(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [("player-1", "Trade Target", "Trade", "Target")],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?)",
        [
            ("NYA_2023", "NYA", "Historical Yankees"),
            ("NYA_2024", "NYA", "New York Yankees"),
            ("CHN_2024", "CHN", "Chicago Cubs"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "player-1",
                2024,
                "NYA",
                "batter",
                100,
                10,
                20,
                0.300,
                1.0,
                10.0,
                5.00,
                6.00,
                0.2,
                1.2,
                1_000_000,
                2_000_000,
                "overpaid",
            ),
            (
                "player-1",
                2024,
                "CHN",
                "pitcher",
                30,
                5,
                4,
                0.500,
                0.8,
                90.0,
                3.00,
                2.00,
                2.0,
                2.8,
                3_000_000,
                4_000_000,
                "surplus_value",
            ),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-1"
    assert row["year_id"] == 2024
    assert row["team_id"] == "CHN"
    assert row["team_name"] == "Chicago Cubs"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 130
    assert row["hr"] == 15
    assert row["bb"] == 24
    assert row["ip"] == pytest.approx(100.0)
    assert row["woba"] == pytest.approx(((0.300 * 100) + (0.500 * 30)) / 130)
    assert row["fip"] == pytest.approx(((5.00 * 10) + (3.00 * 90)) / 100)
    assert row["era"] == pytest.approx(((6.00 * 10) + (2.00 * 90)) / 100)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(4_000_000)
    assert row["surplus_value"] == pytest.approx(6_000_000)
    assert row["contract_label"] == "surplus_value"


def test_player_query_preserves_same_name_players_by_player_id(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("alex-1", "Alex Gonzalez", "Alex", "Gonzalez"),
            ("alex-2", "Alex Gonzalez", "Alex", "Gonzalez"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?)",
        [("TOR_2001", "TOR", "Toronto Blue Jays")],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("alex-1", 2001, "TOR", "batter", 500, 20, 45, 0.340, 3.1, 0, None, None, 0, 3.1, 5_000_000, 19_800_000, "surplus_value"),
            ("alex-2", 2001, "TOR", "batter", 300, 8, 20, 0.290, 0.9, 0, None, None, 0, 0.9, 1_000_000, 6_200_000, "fair_value"),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 2
    assert set(result["player_id"]) == {"alex-1", "alex-2"}
    assert result["name_full"].tolist() == ["Alex Gonzalez", "Alex Gonzalez"]
