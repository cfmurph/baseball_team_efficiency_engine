from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture()
def player_metrics_con() -> duckdb.DuckDBPyConnection:
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
            pa INTEGER,
            hr INTEGER,
            bb INTEGER,
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
            ("traded001", "Taylor Traded", "Taylor", "Traded"),
            ("same001", "Chris Same", "Chris", "Same"),
            ("same002", "Chris Same", "Chris", "Same"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?)",
        [
            ("BOS_2023", "BOS", "Boston 2023"),
            ("BOS_2024", "BOS", "Boston 2024"),
            ("LAD_2024", "LAD", "Los Angeles"),
            ("NYY_2024", "NYY", "New York"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "traded001",
                2024,
                "BOS",
                "batter",
                100,
                5,
                10,
                0.300,
                1.0,
                0.0,
                None,
                None,
                0.0,
                1.0,
                1_000_000,
                7_000_000,
                "surplus_value",
            ),
            (
                "traded001",
                2024,
                "LAD",
                "batter",
                300,
                15,
                30,
                0.400,
                3.0,
                0.0,
                None,
                None,
                0.0,
                3.0,
                2_000_000,
                22_000_000,
                "fair_value",
            ),
            (
                "same001",
                2024,
                "NYY",
                "pitcher",
                0,
                0,
                0,
                None,
                0.0,
                80.0,
                3.50,
                4.00,
                2.0,
                2.0,
                5_000_000,
                11_000_000,
                "fair_value",
            ),
            (
                "same002",
                2024,
                "NYY",
                "pitcher",
                0,
                0,
                0,
                None,
                0.0,
                60.0,
                4.50,
                5.00,
                1.0,
                1.0,
                4_000_000,
                4_000_000,
                "overpaid",
            ),
        ],
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_collapses_traded_player_to_one_weighted_season_row(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    df = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    traded = df[df["player_id"] == "traded001"]

    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "LAD"
    assert row["team_name"] == "Los Angeles"
    assert row["pa"] == 400
    assert row["hr"] == 20
    assert row["bb"] == 40
    assert row["woba"] == pytest.approx(0.375)
    assert row["batting_war"] == pytest.approx(4.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000)
    assert row["surplus_value"] == pytest.approx(29_000_000)
    assert row["contract_label"] == "fair_value"


def test_player_query_preserves_distinct_same_name_players(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    df = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    same_name = df[df["name_full"] == "Chris Same"].sort_values("player_id")

    assert same_name["player_id"].tolist() == ["same001", "same002"]
    assert same_name["player_war"].tolist() == [pytest.approx(2.0), pytest.approx(1.0)]


def test_player_query_joins_team_name_by_season_specific_team_key(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    df = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    boston_row = df[df["player_id"] == "traded001"].iloc[0]

    assert boston_row["team_name"] != "Boston 2023"
