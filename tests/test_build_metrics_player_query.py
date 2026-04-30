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
            name_first VARCHAR,
            name_last VARCHAR,
            name_full VARCHAR
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


def test_player_query_aggregates_traded_player_to_one_season_row(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('player-a', 'Pat', 'Example', 'Pat Example')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2024', 'AAA', 'Alpha Club'),
            ('BBB_2024', 'BBB', 'Beta Club')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('player-a', 2024, 'AAA', 'batter', 100, 5, 10, 0.310, 1.0, NULL, NULL, NULL, 0.0, 1.0, 1000000, 7000000, 'surplus_value'),
            ('player-a', 2024, 'BBB', 'batter', 200, 8, 20, 0.350, 2.5, NULL, NULL, NULL, 0.0, 2.5, 2000000, 18000000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-a"
    assert row["year_id"] == 2024
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Club"
    assert row["pa"] == 300
    assert row["hr"] == 13
    assert row["bb"] == 30
    assert row["player_war"] == pytest.approx(3.5)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 25_000_000


def test_player_query_uses_season_specific_team_name_without_historical_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('player-a', 'Pat', 'Example', 'Pat Example')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2023', 'AAA', 'Old Alpha Name'),
            ('AAA_2024', 'AAA', 'Current Alpha Name')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('player-a', 2024, 'AAA', 'batter', 100, 5, 10, 0.310, 1.0, NULL, NULL, NULL, 0.0, 1.0, 1000000, 7000000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Current Alpha Name"
    assert row["pa"] == 100
    assert row["player_war"] == pytest.approx(1.0)
