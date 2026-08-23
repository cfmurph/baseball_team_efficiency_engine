from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _register_player_query_tables(con: duckdb.DuckDBPyConnection) -> None:
    player_rows = pd.DataFrame(
        [
            {
                "player_id": "traded01",
                "season_key": 2020,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 100,
                "hr": 10,
                "bb": 20,
                "woba": 0.350,
                "batting_war": 1.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "war_source": "approx",
                "salary": 1_000_000,
                "surplus_value": 11_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "traded01",
                "season_key": 2020,
                "team_id": "BBB",
                "player_type": "batter",
                "pa": 50,
                "hr": 4,
                "bb": 5,
                "woba": 0.300,
                "batting_war": 0.5,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.5,
                "war_source": "approx",
                "salary": 500_000,
                "surplus_value": 3_500_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "twoway01",
                "season_key": 2020,
                "team_id": "AAA",
                "player_type": "pitcher",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 10.0,
                "fip": 3.00,
                "era": 2.70,
                "pitching_war": 0.7,
                "player_war": 0.7,
                "war_source": "approx",
                "salary": 900_000,
                "surplus_value": 4_700_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "twoway01",
                "season_key": 2020,
                "team_id": "BBB",
                "player_type": "both",
                "pa": 30,
                "hr": 1,
                "bb": 2,
                "woba": 0.320,
                "batting_war": 0.2,
                "ip": 5.0,
                "fip": 4.00,
                "era": 3.60,
                "pitching_war": 0.1,
                "player_war": 0.3,
                "war_source": "approx",
                "salary": 100_000,
                "surplus_value": 2_300_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    player_rows["ip"] = player_rows["ip"].astype("float64")
    player_rows["fip"] = player_rows["fip"].astype("float64")
    player_rows["era"] = player_rows["era"].astype("float64")
    player_rows["woba"] = player_rows["woba"].astype("float64")

    dim_player = pd.DataFrame(
        [
            {"player_id": "traded01", "name_full": "Trade Target", "name_first": "Trade", "name_last": "Target"},
            {"player_id": "twoway01", "name_full": "Two Way", "name_first": "Two", "name_last": "Way"},
        ]
    )

    # Historical team names should not fan out a player season; only the
    # matching team_key row should join to a stint.
    dim_team = pd.DataFrame(
        [
            {"team_key": "AAA_2019", "team_id": "AAA", "team_name": "Old Alpha"},
            {"team_key": "AAA_2020", "team_id": "AAA", "team_name": "Alpha Aces"},
            {"team_key": "BBB_2019", "team_id": "BBB", "team_name": "Old Beta"},
            {"team_key": "BBB_2020", "team_id": "BBB", "team_name": "Beta Bears"},
        ]
    )

    con.register("player_rows", player_rows)
    con.register("dim_player_rows", dim_player)
    con.register("dim_team_rows", dim_team)
    con.execute("CREATE TABLE fact_player_season AS SELECT * FROM player_rows")
    con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_rows")
    con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_rows")


def test_player_query_returns_one_row_per_player_season_without_team_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert set(result["player_id"]) == {"traded01", "twoway01"}
    assert result.duplicated(["player_id", "year_id"]).sum() == 0


def test_player_query_aggregates_traded_player_stints_and_keeps_primary_team() -> None:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    traded = result.set_index("player_id").loc["traded01"]
    assert traded["team_id"] == "AAA"
    assert traded["team_name"] == "Alpha Aces"
    assert traded["player_type"] == "batter"
    assert traded["pa"] == 150
    assert traded["hr"] == 14
    assert traded["bb"] == 25
    assert traded["player_war"] == 2.0
    assert traded["salary"] == 1_500_000
    assert traded["surplus_value"] == 14_500_000
    assert traded["contract_label"] == "surplus_value"


def test_player_query_preserves_both_player_type_and_pitching_rates() -> None:
    con = duckdb.connect(":memory:")
    try:
        _register_player_query_tables(con)

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    two_way = result.set_index("player_id").loc["twoway01"]
    assert two_way["team_id"] == "AAA"
    assert two_way["player_type"] == "both"
    assert two_way["pa"] == 30
    assert two_way["ip"] == 15.0
    assert two_way["fip"] == pytest.approx(10 / 3)
    assert two_way["era"] == pytest.approx(3.0)
    assert two_way["pitching_war"] == pytest.approx(0.8)
    assert two_way["player_war"] == 1.0


def test_player_query_preserves_distinct_same_name_players() -> None:
    """Two people who share a display name must stay on separate export rows."""
    con = duckdb.connect(":memory:")
    con.register(
        "fact_player_season",
        pd.DataFrame(
            [
                {
                    "player_id": "same001",
                    "season_key": 2024,
                    "team_id": "NYY",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 80.0,
                    "fip": 3.50,
                    "era": 4.00,
                    "pitching_war": 2.0,
                    "player_war": 2.0,
                    "war_source": "approx",
                    "salary": 5_000_000,
                    "surplus_value": 11_000_000,
                    "contract_label": "fair_value",
                },
                {
                    "player_id": "same002",
                    "season_key": 2024,
                    "team_id": "NYY",
                    "player_type": "pitcher",
                    "pa": 0,
                    "hr": 0,
                    "bb": 0,
                    "woba": None,
                    "batting_war": 0.0,
                    "ip": 60.0,
                    "fip": 4.50,
                    "era": 5.00,
                    "pitching_war": 1.0,
                    "player_war": 1.0,
                    "war_source": "approx",
                    "salary": 4_000_000,
                    "surplus_value": 4_000_000,
                    "contract_label": "overpaid",
                },
            ]
        ),
    )
    con.register(
        "dim_player",
        pd.DataFrame(
            [
                {
                    "player_id": "same001",
                    "name_full": "Chris Same",
                    "name_first": "Chris",
                    "name_last": "Same",
                },
                {
                    "player_id": "same002",
                    "name_full": "Chris Same",
                    "name_first": "Chris",
                    "name_last": "Same",
                },
            ]
        ),
    )
    con.register(
        "dim_team",
        pd.DataFrame(
            [
                {"team_key": "NYY_2024", "team_id": "NYY", "team_name": "New York"},
            ]
        ),
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()
    same_name = result[result["name_full"] == "Chris Same"].sort_values("player_id")

    assert same_name["player_id"].tolist() == ["same001", "same002"]
    assert same_name["player_war"].tolist() == [pytest.approx(2.0), pytest.approx(1.0)]
