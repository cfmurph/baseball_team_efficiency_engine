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
            franchise_id VARCHAR,
            team_name VARCHAR,
            league_id VARCHAR
        )
        """
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_sums_traded_player_stints_and_uses_highest_war_team(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [("traded01", "Traded Player", "Traded", "Player")],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("AAA_2012", "AAA", "AAA", "Alpha Club", "AL"),
            ("BBB_2012", "BBB", "BBB", "Beta Club", "NL"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("traded01", 2012, "AAA", "batter", 100, 5, 10, 0.330, 1.0, None, None, None, 0.0, 1.0, 1_000_000, 6_000_000, "surplus_value"),
            ("traded01", 2012, "BBB", "batter", 150, 8, 15, 0.350, 3.0, None, None, None, 0.0, 3.0, 2_000_000, 19_000_000, "surplus_value"),
        ],
    )

    row = con.execute(_PLAYER_QUERY).fetchdf().iloc[0]

    assert row["player_id"] == "traded01"
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Club"
    assert row["pa"] == 250
    assert row["hr"] == 13
    assert row["bb"] == 25
    assert row["batting_war"] == 4.0
    assert row["player_war"] == 4.0
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 25_000_000


def test_player_query_uses_season_specific_team_name_without_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [("rename01", "Renamed Team Player", "Renamed", "Player")],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("MON_2004", "MON", "WSN", "Montreal Expos", "NL"),
            ("MON_2005", "MON", "WSN", "Washington Nationals", "NL"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("rename01", 2004, "MON", "batter", 100, 4, 12, 0.320, 2.0, None, None, None, 0.0, 2.0, 1_500_000, 12_500_000, "surplus_value"),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Montreal Expos"
    assert row["pa"] == 100
    assert row["hr"] == 4
    assert row["bb"] == 12
    assert row["batting_war"] == 2.0
    assert row["player_war"] == 2.0
    assert row["salary"] == 1_500_000
    assert row["surplus_value"] == 12_500_000
