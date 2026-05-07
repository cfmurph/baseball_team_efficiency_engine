from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def player_metrics_connection() -> duckdb.DuckDBPyConnection:
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

    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("trade001", "Taylor Traded", "Taylor", "Traded"),
            ("smith001", "Alex Smith", "Alex", "Smith"),
            ("smith002", "Alex Smith", "Alex", "Smith"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?)",
        [
            ("AAA", "Aces"),
            ("AAA", "Aces"),
            ("BBB", "Bears"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("trade001", 2024, "AAA", "batter", 100, 4, 10, 0.310, 0.7, 0.0, None, None, 0.0, 0.7, 1_000_000, 5_000_000, "surplus"),
            ("trade001", 2024, "BBB", "batter", 200, 8, 20, 0.370, 2.3, 0.0, None, None, 0.0, 2.3, 2_000_000, 8_000_000, "surplus"),
            ("smith001", 2024, "AAA", "batter", 50, 1, 5, 0.290, 0.2, 0.0, None, None, 0.0, 0.2, 750_000, 1_000_000, "minimum"),
            ("smith002", 2024, "AAA", "batter", 75, 2, 8, 0.320, 0.6, 0.0, None, None, 0.0, 0.6, 800_000, 2_000_000, "minimum"),
        ],
    )

    try:
        yield con
    finally:
        con.close()


def test_player_query_aggregates_traded_stints_without_team_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_connection.execute(_PLAYER_QUERY).fetchdf()

    traded = result[result["player_id"] == "trade001"].iloc[0]

    assert not result.duplicated(["player_id", "year_id"]).any()
    assert len(result) == 3
    assert traded["year_id"] == 2024
    assert traded["team_id"] == "BBB"
    assert traded["team_name"] == "Bears"
    assert traded["pa"] == 300
    assert traded["hr"] == 12
    assert traded["bb"] == 30
    assert traded["player_war"] == pytest.approx(3.0)
    assert traded["salary"] == pytest.approx(3_000_000)
    assert traded["surplus_value"] == pytest.approx(13_000_000)


def test_player_query_preserves_same_name_players_as_distinct_player_ids(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_connection.execute(_PLAYER_QUERY).fetchdf()

    alex_smiths = result[result["name_full"] == "Alex Smith"].sort_values("player_id")

    assert alex_smiths["player_id"].tolist() == ["smith001", "smith002"]
    assert alex_smiths["pa"].tolist() == [50, 75]
