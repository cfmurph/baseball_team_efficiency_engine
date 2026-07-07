from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture
def metrics_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)

    con.executemany(
        """
        INSERT INTO dim_team (team_key, team_id, franchise_id, team_name, league_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("NYA_2010", "NYA", "NYY", "New York Yankees", "AL"),
            ("NYA_2011", "NYA", "NYY", "New York Yankees", "AL"),
            ("NYA_2012", "NYA", "NYY", "New York Yankees", "AL"),
            ("BOS_2010", "BOS", "BRS", "Boston Red Sox", "AL"),
        ],
    )
    con.executemany(
        """
        INSERT INTO dim_player (
            player_id, name_first, name_last, name_full,
            birth_year, birth_country, throws, bats
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("traded001", "Trade", "Star", "Trade Star", 1980, "USA", "R", "R"),
            ("single001", "Single", "Season", "Single Season", 1981, "USA", "L", "L"),
            ("alex001", "Alex", "Gonzalez", "Alex Gonzalez", 1977, "USA", "R", "R"),
            ("alex002", "Alex", "Gonzalez", "Alex Gonzalez", 1973, "VEN", "R", "R"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season (
            player_id, season_key, team_id, player_type,
            pa, hr, bb, woba, batting_war,
            ip, fip, era, pitching_war,
            player_war, salary, surplus_value, contract_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "traded001", 2010, "BOS", "batter",
                100, 8, 14, 0.330, 1.0,
                0, None, None, 0,
                1.0, 1_000_000, 9_000_000, "value",
            ),
            (
                "traded001", 2010, "NYA", "pitcher",
                25, 2, 4, 0.350, 0.5,
                50, 3.10, 2.75, 4.5,
                5.0, 3_000_000, 20_000_000, "star",
            ),
            (
                "single001", 2010, "NYA", "batter",
                500, 30, 60, 0.380, 2.0,
                0, None, None, 0,
                2.0, 2_000_000, 18_000_000, "value",
            ),
            (
                "alex001", 2010, "BOS", "batter",
                300, 10, 30, 0.310, 1.2,
                0, None, None, 0,
                1.2, 750_000, 6_000_000, "value",
            ),
            (
                "alex002", 2010, "BOS", "batter",
                250, 5, 20, 0.295, 0.8,
                0, None, None, 0,
                0.8, 700_000, 4_000_000, "value",
            ),
        ],
    )

    yield con
    con.close()


def _player_metrics(con: duckdb.DuckDBPyConnection):
    return con.execute(_PLAYER_QUERY).fetchdf()


def test_player_query_emits_unique_player_season_rows(metrics_db):
    df = _player_metrics(metrics_db)

    assert not df.duplicated(["player_id", "year_id"]).any()


def test_player_query_does_not_fan_out_on_multi_season_dim_team(metrics_db):
    df = _player_metrics(metrics_db)

    row = df[df["player_id"] == "single001"].iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["player_war"] == pytest.approx(2.0)
    assert row["salary"] == pytest.approx(2_000_000)
    assert row["surplus_value"] == pytest.approx(18_000_000)


def test_player_query_aggregates_traded_player_to_primary_team(metrics_db):
    df = _player_metrics(metrics_db)

    row = df[df["player_id"] == "traded001"].iloc[0]
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "New York Yankees"
    assert row["contract_label"] == "star"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == pytest.approx(125)
    assert row["hr"] == pytest.approx(10)
    assert row["bb"] == pytest.approx(18)
    assert row["batting_war"] == pytest.approx(1.5)
    assert row["ip"] == pytest.approx(50)
    assert row["pitching_war"] == pytest.approx(4.5)
    assert row["player_war"] == pytest.approx(6.0)
    assert row["salary"] == pytest.approx(4_000_000)
    assert row["surplus_value"] == pytest.approx(29_000_000)


def test_player_query_keeps_same_name_players_distinct(metrics_db):
    df = _player_metrics(metrics_db)

    same_name = df[df["name_full"] == "Alex Gonzalez"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["alex001", "alex002"]
    assert same_name["year_id"].tolist() == [2010, 2010]
