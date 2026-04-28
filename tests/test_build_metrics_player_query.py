from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _register_player_query_tables(con: duckdb.DuckDBPyConnection) -> None:
    """Create the minimal warehouse tables needed by _PLAYER_QUERY."""
    fact_player_season = pd.DataFrame(
        [
            {
                "player_id": "traded-player",
                "season_key": 2024,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 100,
                "hr": 5,
                "bb": 10,
                "woba": 0.300,
                "batting_war": 1.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.0,
                "salary": 1_000_000,
                "surplus_value": 4_000_000,
                "contract_label": "value",
            },
            {
                "player_id": "traded-player",
                "season_key": 2024,
                "team_id": "BBB",
                "player_type": "batter",
                "pa": 40,
                "hr": 3,
                "bb": 4,
                "woba": 0.360,
                "batting_war": 2.5,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.5,
                "salary": 2_000_000,
                "surplus_value": 8_000_000,
                "contract_label": "arbitration",
            },
            {
                "player_id": "two-way-player",
                "season_key": 2024,
                "team_id": "AAA",
                "player_type": "both",
                "pa": 10,
                "hr": 1,
                "bb": 1,
                "woba": 0.500,
                "batting_war": 0.4,
                "ip": 12.0,
                "fip": 3.20,
                "era": 3.50,
                "pitching_war": 0.8,
                "player_war": 1.2,
                "salary": 750_000,
                "surplus_value": 2_000_000,
                "contract_label": "pre_arbitration",
            },
        ]
    )
    dim_player = pd.DataFrame(
        [
            {
                "player_id": "traded-player",
                "name_full": "Traded Player",
                "name_first": "Traded",
                "name_last": "Player",
            },
            {
                "player_id": "two-way-player",
                "name_full": "Two Way",
                "name_first": "Two",
                "name_last": "Way",
            },
        ]
    )
    dim_team = pd.DataFrame(
        [
            {"team_id": "AAA", "team_name": "Alpha Club"},
            {"team_id": "AAA", "team_name": "Alpha Club"},
            {"team_id": "BBB", "team_name": "Beta Club"},
            {"team_id": "BBB", "team_name": "Beta Club"},
        ]
    )

    con.register("fact_player_season_df", fact_player_season)
    con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
    con.register("dim_player_df", dim_player)
    con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
    con.register("dim_team_df", dim_team)
    con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")


@pytest.fixture
def player_query_df() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def test_player_query_keeps_one_row_per_player_season(player_query_df: pd.DataFrame) -> None:
    keys = player_query_df[["player_id", "year_id"]]

    assert len(player_query_df) == 2
    assert not keys.duplicated().any()
    assert set(player_query_df["player_id"]) == {"traded-player", "two-way-player"}


def test_player_query_consolidates_traded_player_stints(player_query_df: pd.DataFrame) -> None:
    traded = player_query_df.set_index("player_id").loc["traded-player"]

    assert traded["year_id"] == 2024
    assert traded["team_id"] == "BBB"
    assert traded["team_name"] == "Beta Club"
    assert traded["player_type"] == "batter"
    assert traded["contract_label"] == "arbitration"

    assert traded["pa"] == 140
    assert traded["hr"] == 8
    assert traded["bb"] == 14
    assert traded["batting_war"] == pytest.approx(3.5)
    assert traded["player_war"] == pytest.approx(3.5)
    assert traded["salary"] == 3_000_000
    assert traded["surplus_value"] == 12_000_000


def test_player_query_preserves_two_way_player_rates(player_query_df: pd.DataFrame) -> None:
    two_way = player_query_df.set_index("player_id").loc["two-way-player"]

    assert two_way["player_type"] == "both"
    assert two_way["woba"] == pytest.approx(0.500)
    assert two_way["fip"] == pytest.approx(3.20)
    assert two_way["era"] == pytest.approx(3.50)
    assert two_way["pitching_war"] == pytest.approx(0.8)
