from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _register_player_query_tables(con: duckdb.DuckDBPyConnection) -> None:
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


def test_player_query_consolidates_traded_player_without_team_year_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [
                ("traded", "Taylor Traded", "Taylor", "Traded"),
                ("pitcher", "Pat Pitcher", "Pat", "Pitcher"),
            ],
        )
        con.executemany(
            "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
            [
                ("AAA_2023", "AAA", "AAA", "Old Alpha", "AL"),
                ("AAA_2024", "AAA", "AAA", "Alpha", "AL"),
                ("BBB_2024", "BBB", "BBB", "Beta", "NL"),
                ("BBB_2025", "BBB", "BBB", "Future Beta", "NL"),
            ],
        )
        con.executemany(
            "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("traded", 2024, "AAA", "batter", 100, 10, 10, 0.500, 2.0, 0, None, None, 0.0, 2.0, 100, 20, "star"),
                ("traded", 2024, "BBB", "batter", 300, 15, 30, 0.300, 1.0, 0, None, None, 0.0, 1.0, 200, 30, "regular"),
                ("pitcher", 2024, "AAA", "pitcher", 0, 0, 0, None, 0.0, 10, 1.00, 2.00, 0.5, 0.5, 50, 5, "regular"),
                ("pitcher", 2024, "BBB", "pitcher", 0, 0, 0, None, 0.0, 30, 5.00, 6.00, 1.5, 1.5, 75, 8, "star"),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert set(result["player_id"]) == {"traded", "pitcher"}

    traded = result[result["player_id"] == "traded"].iloc[0]
    assert traded["team_id"] == "AAA"
    assert traded["team_name"] == "Alpha"
    assert traded["player_type"] == "batter"
    assert traded["pa"] == 400
    assert traded["hr"] == 25
    assert traded["bb"] == 40
    assert traded["woba"] == pytest.approx(0.350)
    assert traded["batting_war"] == pytest.approx(3.0)
    assert traded["player_war"] == pytest.approx(3.0)
    assert traded["salary"] == pytest.approx(300)
    assert traded["surplus_value"] == pytest.approx(50)
    assert traded["contract_label"] == "star"

    pitcher = result[result["player_id"] == "pitcher"].iloc[0]
    assert pitcher["team_id"] == "BBB"
    assert pitcher["team_name"] == "Beta"
    assert pitcher["player_type"] == "pitcher"
    assert pitcher["ip"] == 40
    assert pitcher["fip"] == pytest.approx(4.00)
    assert pitcher["era"] == pytest.approx(5.00)
    assert pitcher["pitching_war"] == pytest.approx(2.0)
    assert pitcher["player_war"] == pytest.approx(2.0)
    assert pitcher["contract_label"] == "star"


def test_player_query_preserves_distinct_players_with_same_name() -> None:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)
        con.executemany(
            "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
            [
                ("smith-a", "Jordan Smith", "Jordan", "Smith"),
                ("smith-b", "Jordan Smith", "Jordan", "Smith"),
            ],
        )
        con.executemany(
            "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
            [
                ("AAA_2024", "AAA", "AAA", "Alpha", "AL"),
                ("BBB_2024", "BBB", "BBB", "Beta", "NL"),
            ],
        )
        con.executemany(
            "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                ("smith-a", 2024, "AAA", "batter", 200, 8, 20, 0.320, 1.0, 0, None, None, 0.0, 1.0, 100, 10, "regular"),
                ("smith-b", 2024, "BBB", "pitcher", 0, 0, 0, None, 0.0, 50, 3.50, 4.00, 2.0, 2.0, 150, 15, "star"),
            ],
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert result["name_full"].tolist().count("Jordan Smith") == 2
    assert set(result["player_id"]) == {"smith-a", "smith-b"}
