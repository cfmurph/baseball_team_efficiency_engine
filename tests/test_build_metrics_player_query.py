from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_player_metric_tables(con: duckdb.DuckDBPyConnection) -> None:
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
            franchise_id VARCHAR,
            team_name VARCHAR,
            league_id VARCHAR
        )
        """
    )


def test_player_query_aggregates_traded_player_without_team_name_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        _create_player_metric_tables(con)
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [("traded001", "Alex", "Example", "Alex Example")],
        )
        con.executemany(
            "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
            [
                ("OAK_2000", "OAK", "ATH", "Oakland Athletics", "AL"),
                ("OAK_2001", "OAK", "ATH", "Oakland A's", "AL"),
                ("NYA_2000", "NYA", "NYY", "New York Yankees", "AL"),
            ],
        )
        con.executemany(
            """
            INSERT INTO fact_player_season VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "traded001",
                    2000,
                    "OAK",
                    "batter",
                    100,
                    5,
                    10,
                    0.300,
                    1.0,
                    10,
                    2.0,
                    1.0,
                    0.5,
                    1.5,
                    1_000_000,
                    11_000_000,
                    "surplus_value",
                ),
                (
                    "traded001",
                    2000,
                    "NYA",
                    "both",
                    300,
                    15,
                    30,
                    0.400,
                    3.0,
                    30,
                    4.0,
                    3.0,
                    1.0,
                    4.0,
                    2_000_000,
                    30_000_000,
                    "fair_value",
                ),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "New York Yankees"
    assert row["player_type"] == "both"
    assert row["pa"] == 400
    assert row["hr"] == 20
    assert row["bb"] == 40
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 41_000_000
    assert row["contract_label"] == "fair_value"
    assert row["woba"] == pytest.approx(0.375)
    assert row["fip"] == pytest.approx(3.5)
    assert row["era"] == pytest.approx(2.5)
    assert row["player_war"] == pytest.approx(5.5)


def test_player_query_preserves_distinct_players_with_same_display_name() -> None:
    con = duckdb.connect(":memory:")
    try:
        _create_player_metric_tables(con)
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [
                ("smith001", "Chris", "Smith", "Chris Smith"),
                ("smith002", "Chris", "Smith", "Chris Smith"),
            ],
        )
        con.executemany(
            "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
            [("BOS_2004", "BOS", "BOS", "Boston Red Sox", "AL")],
        )
        con.executemany(
            """
            INSERT INTO fact_player_season VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    "smith001",
                    2004,
                    "BOS",
                    "batter",
                    250,
                    12,
                    20,
                    0.350,
                    2.0,
                    0,
                    None,
                    None,
                    0.0,
                    2.0,
                    500_000,
                    15_500_000,
                    "surplus_value",
                ),
                (
                    "smith002",
                    2004,
                    "BOS",
                    "pitcher",
                    0,
                    0,
                    0,
                    None,
                    0.0,
                    80,
                    3.25,
                    3.80,
                    1.5,
                    1.5,
                    750_000,
                    11_250_000,
                    "fair_value",
                ),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert set(result["player_id"]) == {"smith001", "smith002"}
    assert result["name_full"].tolist() == ["Chris Smith", "Chris Smith"]
    assert result.set_index("player_id").loc["smith001", "pa"] == 250
    assert result.set_index("player_id").loc["smith002", "ip"] == 80
