from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def player_query_connection() -> duckdb.DuckDBPyConnection:
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


def test_player_query_aggregates_traded_stints_without_team_dimension_fanout(
    player_query_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_query_connection
    con.execute("INSERT INTO dim_player VALUES ('smitha01', 'Alex Smith', 'Alex', 'Smith')")
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYA_2020', 'NYA', 'New York Yankees'),
            ('NYA_2021', 'NYA', 'New York Yankees'),
            ('NYA_2022', 'NYA', 'Renamed Yankees'),
            ('BOS_2020', 'BOS', 'Boston Red Sox')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('smitha01', 2020, 'NYA', 'batter', 100, 10, 20, 0.400, 1.0,
             10, 2.00, 3.00, 0.5, 1.0, 1000000, 7000000, 'surplus_value'),
            ('smitha01', 2020, 'BOS', 'pitcher', 300, 5, 30, 0.300, 0.5,
             30, 4.00, 5.00, 2.5, 3.0, 2000000, 22000000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "smitha01"
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == pytest.approx(400)
    assert row["salary"] == pytest.approx(3000000)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["woba"] == pytest.approx(0.325)
    assert row["fip"] == pytest.approx(3.5)
    assert row["era"] == pytest.approx(4.5)


def test_player_query_preserves_same_name_players_by_player_id(
    player_query_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_query_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('smitha01', 'Alex Smith', 'Alex', 'Smith'),
            ('smithb01', 'Alex Smith', 'Alex', 'Smith')
        """
    )
    con.execute("INSERT INTO dim_team VALUES ('NYA_2020', 'NYA', 'New York Yankees')")
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('smitha01', 2020, 'NYA', 'batter', 200, 12, 30, 0.350, 2.0,
             0, NULL, NULL, 0, 2.0, 1000000, 15000000, 'surplus_value'),
            ('smithb01', 2020, 'NYA', 'batter', 100, 2, 10, 0.300, 0.5,
             0, NULL, NULL, 0, 0.5, 750000, 3250000, 'fair_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert set(result["player_id"]) == {"smitha01", "smithb01"}
    assert result["name_full"].tolist() == ["Alex Smith", "Alex Smith"]
