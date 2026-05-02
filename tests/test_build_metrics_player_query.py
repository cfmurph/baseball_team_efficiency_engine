from __future__ import annotations

import duckdb
import pandas as pd
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
            salary INTEGER,
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
            team_name VARCHAR
        )
        """
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_collapses_traded_player_to_one_weighted_season(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('traded', 'Trade Deadline', 'Trade', 'Deadline')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2024', 'AAA', 'Alpha Aces'),
            ('BBB_2024', 'BBB', 'Beta Bears')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('traded', 2024, 'AAA', 'batter', 100, 10, 20, 0.300, 1.0,
             10.0, 4.00, 3.00, 0.5, 1.5, 1000000, 11000000, 'value'),
            ('traded', 2024, 'BBB', 'pitcher', 300, 20, 40, 0.400, 2.0,
             30.0, 2.00, 1.00, 1.5, 3.5, 2000000, 28000000, 'star')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded"
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Bears"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 400
    assert row["hr"] == 30
    assert row["bb"] == 60
    assert row["woba"] == pytest.approx(0.375)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(2.5)
    assert row["era"] == pytest.approx(1.5)
    assert row["player_war"] == pytest.approx(5.0)
    assert row["salary"] == 3000000
    assert row["surplus_value"] == pytest.approx(39000000)
    assert row["contract_label"] == "star"


def test_player_query_joins_team_name_by_season_key_without_historical_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('stable', 'Stable Veteran', 'Stable', 'Veteran')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2023', 'AAA', 'Old Alpha Name'),
            ('AAA_2024', 'AAA', 'Alpha Aces')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('stable', 2024, 'AAA', 'batter', 250, 12, 30, 0.333, 2.5,
             0.0, NULL, NULL, 0.0, 2.5, 1500000, 18000000, 'value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert result[["player_id", "year_id"]].duplicated().sum() == 0
    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Alpha Aces"
    assert row["pa"] == 250
    assert row["salary"] == 1500000
