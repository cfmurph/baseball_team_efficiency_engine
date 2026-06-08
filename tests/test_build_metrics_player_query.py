from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def player_metrics_db() -> duckdb.DuckDBPyConnection:
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
    yield con
    con.close()


def test_player_query_aggregates_traded_players_without_historical_team_fanout(
    player_metrics_db: duckdb.DuckDBPyConnection,
) -> None:
    player_metrics_db.execute(
        """
        INSERT INTO dim_player VALUES
            ('traded-1', 'Traded Player', 'Traded', 'Player')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYA_1910', 'NYA', 'New York Highlanders'),
            ('NYA_2010', 'NYA', 'New York Yankees'),
            ('BOS_2010', 'BOS', 'Boston Red Sox')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('traded-1', 2010, 'NYA', 'batter', 100, 5, 10, 0.400, 1.0, NULL, NULL, NULL, 0.0, 1.0, 1000000, 7000000, 'surplus_value'),
            ('traded-1', 2010, 'BOS', 'batter', 300, 15, 30, 0.300, 3.0, NULL, NULL, NULL, 0.0, 3.0, 2000000, 22000000, 'surplus_value')
        """
    )

    result = player_metrics_db.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == pytest.approx(400)
    assert row["salary"] == pytest.approx(3_000_000)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["woba"] == pytest.approx(((0.400 * 100) + (0.300 * 300)) / 400)


def test_player_query_weights_pitching_rates_by_innings(
    player_metrics_db: duckdb.DuckDBPyConnection,
) -> None:
    player_metrics_db.execute(
        """
        INSERT INTO dim_player VALUES
            ('pitcher-1', 'Stint Pitcher', 'Stint', 'Pitcher')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYA_2010', 'NYA', 'New York Yankees'),
            ('BOS_2010', 'BOS', 'Boston Red Sox')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('pitcher-1', 2010, 'NYA', 'pitcher', NULL, NULL, NULL, NULL, 0.0, 10, 5.00, 6.00, 0.1, 0.1, 1000000, -200000, 'overpaid'),
            ('pitcher-1', 2010, 'BOS', 'pitcher', NULL, NULL, NULL, NULL, 0.0, 90, 3.00, 2.00, 2.0, 2.0, 2000000, 14000000, 'surplus_value')
        """
    )

    row = player_metrics_db.execute(_PLAYER_QUERY).fetchdf().iloc[0]

    assert row["ip"] == pytest.approx(100)
    assert row["fip"] == pytest.approx(((5.00 * 10) + (3.00 * 90)) / 100)
    assert row["era"] == pytest.approx(((6.00 * 10) + (2.00 * 90)) / 100)


def test_player_query_keeps_same_name_players_as_distinct_rows(
    player_metrics_db: duckdb.DuckDBPyConnection,
) -> None:
    player_metrics_db.execute(
        """
        INSERT INTO dim_player VALUES
            ('smith-a', 'Chris Smith', 'Chris', 'Smith'),
            ('smith-b', 'Chris Smith', 'Chris', 'Smith')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYA_2010', 'NYA', 'New York Yankees')
        """
    )
    player_metrics_db.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('smith-a', 2010, 'NYA', 'batter', 200, 7, 20, 0.330, 1.5, NULL, NULL, NULL, 0.0, 1.5, 500000, 11500000, 'surplus_value'),
            ('smith-b', 2010, 'NYA', 'batter', 120, 3, 12, 0.290, 0.4, NULL, NULL, NULL, 0.0, 0.4, 500000, 2700000, 'fair_value')
        """
    )

    result = player_metrics_db.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 2
    assert set(result["player_id"]) == {"smith-a", "smith-b"}
    assert result["name_full"].tolist() == ["Chris Smith", "Chris Smith"]
