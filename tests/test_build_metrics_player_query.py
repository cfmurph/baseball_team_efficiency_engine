from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_player_query_tables(con: duckdb.DuckDBPyConnection) -> None:
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
            player_war DOUBLE,
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
            salary DOUBLE,
            surplus_value DOUBLE,
            contract_label VARCHAR
        )
        """
    )


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    _create_player_query_tables(connection)
    try:
        yield connection
    finally:
        connection.close()


def test_player_query_aggregates_traded_players_with_weighted_rates(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('trade01', 'Traded Player', 'Traded', 'Player')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2020', 'AAA', 'Alpha Aces'),
            ('BBB_2020', 'BBB', 'Beta Bears')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('trade01', 2020, 'AAA', 2.5, 'batter', 100, 5, 10, 0.300, 1.5, 10.0, 4.00, 3.00, 1.0, 1000000, 9000000, 'surplus_value'),
            ('trade01', 2020, 'BBB', 1.0, 'pitcher', 300, 15, 30, 0.400, 0.5, 30.0, 2.00, 5.00, 0.5, 2000000, 8000000, 'fair_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "AAA"
    assert row["team_name"] == "Alpha Aces"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 400
    assert row["hr"] == 20
    assert row["bb"] == 40
    assert row["woba"] == pytest.approx(0.375)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(2.5)
    assert row["era"] == pytest.approx(4.5)
    assert row["player_war"] == pytest.approx(3.5)
    assert row["salary"] == pytest.approx(3000000)
    assert row["surplus_value"] == pytest.approx(17000000)
    assert row["contract_label"] == "surplus_value"


def test_player_query_uses_season_specific_team_join_without_fanout(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('single01', 'Single Team', 'Single', 'Team')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2020', 'AAA', 'Original Name'),
            ('AAA_2021', 'AAA', 'Renamed Franchise')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('single01', 2020, 'AAA', 1.0, 'batter', 100, 7, 8, 0.350, 1.0, NULL, NULL, NULL, 0.0, 500000, 7500000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Original Name"
    assert row["pa"] == 100
    assert row["hr"] == 7
    assert row["salary"] == pytest.approx(500000)
    assert row["surplus_value"] == pytest.approx(7500000)


def test_player_query_preserves_same_name_players_by_player_id(
    con: duckdb.DuckDBPyConnection,
) -> None:
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('smith01', 'Alex Smith', 'Alex', 'Smith'),
            ('smith02', 'Alex Smith', 'Alex', 'Smith')
        """
    )
    con.execute("INSERT INTO dim_team VALUES ('AAA_2020', 'AAA', 'Alpha Aces')")
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('smith01', 2020, 'AAA', 2.0, 'batter', 200, 10, 20, 0.360, 2.0, NULL, NULL, NULL, 0.0, 1000000, 15000000, 'surplus_value'),
            ('smith02', 2020, 'AAA', 1.0, 'batter', 150, 6, 15, 0.330, 1.0, NULL, NULL, NULL, 0.0, 800000, 7200000, 'fair_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert set(result["player_id"]) == {"smith01", "smith02"}
    assert result["name_full"].tolist() == ["Alex Smith", "Alex Smith"]
