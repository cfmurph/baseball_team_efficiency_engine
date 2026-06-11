from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture
def player_metrics_connection():
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany(
        """
        INSERT INTO dim_player (
            player_id, name_first, name_last, name_full, birth_year, birth_country, throws, bats
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("smithjo01", "John", "Smith", "John Smith", 1990, "USA", "R", "R"),
            ("smithjo02", "John", "Smith", "John Smith", 1992, "USA", "L", "L"),
        ],
    )
    con.executemany(
        """
        INSERT INTO dim_team (team_key, team_id, franchise_id, team_name, league_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("NYA_2020", "NYA", "NYY", "New York Yankees", "AL"),
            ("NYA_2021", "NYA", "NYY", "New York Yankees", "AL"),
            ("BOS_2020", "BOS", "BOS", "Boston Red Sox", "AL"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season (
            player_id, season_key, team_id, player_type,
            pa, hr, bb, woba, batting_war,
            ip, fip, era, pitching_war,
            player_war, salary, surplus_value, contract_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "smithjo01", 2020, "NYA", "batter",
                100, 5, 10, 0.300, 1.0,
                0, None, None, 0,
                1.0, 1_000_000, 7_000_000, "surplus_value",
            ),
            (
                "smithjo01", 2020, "BOS", "batter",
                200, 15, 20, 0.360, 2.0,
                0, None, None, 0,
                2.0, 2_000_000, 14_000_000, "surplus_value",
            ),
            (
                "smithjo02", 2020, "NYA", "pitcher",
                0, 0, 0, None, 0,
                50, 3.20, 3.50, 1.5,
                1.5, 1_500_000, 10_500_000, "fair_value",
            ),
        ],
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_keeps_same_name_players_distinct_and_aggregates_traded_stints(
    player_metrics_connection,
) -> None:
    result = player_metrics_connection.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 2
    assert set(result["player_id"]) == {"smithjo01", "smithjo02"}
    assert not result.duplicated(["player_id", "year_id"]).any()

    traded = result[result["player_id"] == "smithjo01"].iloc[0]
    assert traded["pa"] == 300
    assert traded["hr"] == 20
    assert traded["bb"] == 30
    assert traded["player_war"] == pytest.approx(3.0)
    assert traded["salary"] == pytest.approx(3_000_000)
    assert traded["surplus_value"] == pytest.approx(21_000_000)
    assert traded["team_id"] == "BOS"
    assert traded["team_name"] == "Boston Red Sox"

    same_name_pitcher = result[result["player_id"] == "smithjo02"].iloc[0]
    assert same_name_pitcher["name_full"] == "John Smith"
    assert same_name_pitcher["player_type"] == "pitcher"
