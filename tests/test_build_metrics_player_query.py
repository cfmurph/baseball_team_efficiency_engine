from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_player_metrics_tables(con: duckdb.DuckDBPyConnection) -> None:
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


def test_player_query_consolidates_traded_player_without_team_history_fanout() -> None:
    con = duckdb.connect(":memory:")
    _create_player_metrics_tables(con)

    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("player-1", "Traded Player", "Traded", "Player"),
            ("player-2", "Same Name", "Same", "Name"),
            ("player-3", "Same Name", "Same", "Name"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("ABC_1901", "ABC", "ABC", "Old ABC Name", "AL"),
            ("ABC_2024", "ABC", "ABC", "Current ABC Name", "AL"),
            ("DEF_2024", "DEF", "DEF", "DEF Name", "NL"),
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
                "player-1",
                2024,
                "ABC",
                "batter",
                100.0,
                10.0,
                20.0,
                0.400,
                2.0,
                10.0,
                2.00,
                3.00,
                0.4,
                2.4,
                1_000_000.0,
                18_200_000.0,
                "surplus_value",
            ),
            (
                "player-1",
                2024,
                "DEF",
                "pitcher",
                300.0,
                30.0,
                40.0,
                0.300,
                1.0,
                30.0,
                4.00,
                5.00,
                0.6,
                1.6,
                2_000_000.0,
                10_800_000.0,
                "fair_value",
            ),
            (
                "player-2",
                2024,
                "ABC",
                "batter",
                50.0,
                1.0,
                2.0,
                0.250,
                0.1,
                None,
                None,
                None,
                0.0,
                0.1,
                750_000.0,
                50_000.0,
                "fair_value",
            ),
            (
                "player-3",
                2024,
                "ABC",
                "batter",
                60.0,
                2.0,
                3.0,
                0.260,
                0.2,
                None,
                None,
                None,
                0.0,
                0.2,
                800_000.0,
                800_000.0,
                "fair_value",
            ),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert list(result["player_id"]) == ["player-1", "player-3", "player-2"]

    traded = result[result["player_id"] == "player-1"].iloc[0]
    assert traded["team_id"] == "ABC"
    assert traded["team_name"] == "Current ABC Name"
    assert traded["player_type"] == "pitcher"
    assert traded["pa"] == 400.0
    assert traded["hr"] == 40.0
    assert traded["bb"] == 60.0
    assert traded["woba"] == pytest.approx(0.325)
    assert traded["batting_war"] == 3.0
    assert traded["ip"] == 40.0
    assert traded["fip"] == pytest.approx(3.5)
    assert traded["era"] == pytest.approx(4.5)
    assert traded["pitching_war"] == 1.0
    assert traded["player_war"] == 4.0
    assert traded["salary"] == 3_000_000.0
    assert traded["surplus_value"] == 29_000_000.0
    assert traded["contract_label"] == "surplus_value"
