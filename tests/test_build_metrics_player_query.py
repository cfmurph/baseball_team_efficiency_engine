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
    try:
        yield con
    finally:
        con.close()


def test_player_query_aggregates_traded_player_with_weighted_rates(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('trade01', 'Traded Hitter', 'Traded', 'Hitter'),
            ('pitch01', 'Traded Pitcher', 'Traded', 'Pitcher')
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
            ('trade01', 2024, 'AAA', 'batter', 100, 5, 10, 0.400, 1.0, NULL, NULL, NULL, 0.0, 1.0, 1000000, 7000000, 'fair_value'),
            ('trade01', 2024, 'BBB', 'batter', 300, 15, 30, 0.300, 4.0, NULL, NULL, NULL, 0.0, 4.0, 2000000, 30000000, 'surplus_value'),
            ('pitch01', 2024, 'AAA', 'pitcher', NULL, NULL, NULL, NULL, 0.0, 10, 5.00, 6.00, 1.0, 1.0, 1000000, 7000000, 'fair_value'),
            ('pitch01', 2024, 'BBB', 'pitcher', NULL, NULL, NULL, NULL, 0.0, 30, 3.00, 2.00, 2.0, 2.0, 2000000, 22000000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf().set_index("player_id")

    hitter = result.loc["trade01"]
    assert hitter["team_id"] == "BBB"
    assert hitter["team_name"] == "Beta Club"
    assert hitter["pa"] == pytest.approx(400)
    assert hitter["hr"] == pytest.approx(20)
    assert hitter["bb"] == pytest.approx(40)
    assert hitter["woba"] == pytest.approx((0.400 * 100 + 0.300 * 300) / 400)
    assert hitter["batting_war"] == pytest.approx(5.0)
    assert hitter["player_war"] == pytest.approx(5.0)
    assert hitter["salary"] == pytest.approx(3_000_000)
    assert hitter["surplus_value"] == pytest.approx(37_000_000)
    assert hitter["contract_label"] == "surplus_value"

    pitcher = result.loc["pitch01"]
    assert pitcher["team_id"] == "BBB"
    assert pitcher["team_name"] == "Beta Club"
    assert pitcher["ip"] == pytest.approx(40)
    assert pitcher["fip"] == pytest.approx((5.00 * 10 + 3.00 * 30) / 40)
    assert pitcher["era"] == pytest.approx((6.00 * 10 + 2.00 * 30) / 40)
    assert pitcher["pitching_war"] == pytest.approx(3.0)


def test_player_query_joins_season_specific_team_without_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute("INSERT INTO dim_player VALUES ('legacy01', 'Legacy Player', 'Legacy', 'Player')")
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('AAA_2023', 'AAA', 'Old Alpha Name'),
            ('AAA_2024', 'AAA', 'Alpha Club')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('legacy01', 2024, 'AAA', 'batter', 100, 10, 12, 0.350, 2.5, NULL, NULL, NULL, 0.0, 2.5, 1500000, 18500000, 'surplus_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Alpha Club"
    assert row["pa"] == pytest.approx(100)
    assert row["salary"] == pytest.approx(1_500_000)


def test_player_query_preserves_same_name_players_as_distinct_people(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('alex01', 'Alex Gonzalez', 'Alex', 'Gonzalez'),
            ('alex02', 'Alex Gonzalez', 'Alex', 'Gonzalez')
        """
    )
    con.execute("INSERT INTO dim_team VALUES ('AAA_2024', 'AAA', 'Alpha Club')")
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('alex01', 2024, 'AAA', 'batter', 250, 8, 20, 0.315, 1.2, NULL, NULL, NULL, 0.0, 1.2, 900000, 8700000, 'fair_value'),
            ('alex02', 2024, 'AAA', 'batter', 180, 4, 12, 0.285, 0.6, NULL, NULL, NULL, 0.0, 0.6, 800000, 4000000, 'fair_value')
        """
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert set(result["player_id"]) == {"alex01", "alex02"}
    assert result["name_full"].tolist() == ["Alex Gonzalez", "Alex Gonzalez"]
    assert result.duplicated(["player_id", "year_id"]).sum() == 0
