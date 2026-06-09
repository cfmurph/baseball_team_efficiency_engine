from __future__ import annotations

import duckdb

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_consolidates_trades_without_team_dimension_fanout() -> None:
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
            team_name VARCHAR,
            franchise_id VARCHAR,
            league_id VARCHAR
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
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("trade01", "Traded Star", "Traded", "Star"),
            ("alex01", "Alex Gonzalez", "Alex", "Gonzalez"),
            ("alex02", "Alex Gonzalez", "Alex", "Gonzalez"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("AAA_2023", "AAA", "Alpha Aces", "AAA", "AL"),
            ("AAA_2024", "AAA", "Alpha Aces", "AAA", "AL"),
            ("BBB_2024", "BBB", "Beta Bears", "BBB", "NL"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            ("trade01", 2024, "AAA", "batter", 100, 10, 12, 0.400, 1.0, 0, None, None, 0, 1.0, 1_000_000, 8_000_000, "value"),
            ("trade01", 2024, "BBB", "both", 50, 5, 6, 0.300, 0.5, 20, 3.2, 3.8, 1.5, 2.0, 2_000_000, 12_000_000, "star"),
            ("alex01", 2024, "AAA", "batter", 300, 8, 30, 0.330, 2.5, 0, None, None, 0, 2.5, 750_000, 9_000_000, "value"),
            ("alex02", 2024, "AAA", "batter", 250, 6, 24, 0.310, 1.5, 0, None, None, 0, 1.5, 800_000, 4_000_000, "value"),
        ],
    )

    rows = con.execute(_PLAYER_QUERY).fetchdf()

    assert rows[["player_id", "year_id"]].duplicated().sum() == 0
    assert set(rows.loc[rows["name_full"] == "Alex Gonzalez", "player_id"]) == {"alex01", "alex02"}

    traded = rows.set_index("player_id").loc["trade01"]
    assert traded["team_id"] == "BBB"
    assert traded["team_name"] == "Beta Bears"
    assert traded["player_type"] == "both"
    assert traded["pa"] == 150
    assert traded["hr"] == 15
    assert traded["bb"] == 18
    assert traded["ip"] == 20
    assert traded["batting_war"] == 1.5
    assert traded["pitching_war"] == 1.5
    assert traded["player_war"] == 3.0
    assert traded["salary"] == 3_000_000
    assert traded["surplus_value"] == 20_000_000
    assert traded["contract_label"] == "star"
