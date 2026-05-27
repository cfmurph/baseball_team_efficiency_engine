from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _run_player_query(
    fact_player_season: pd.DataFrame,
    dim_team: pd.DataFrame,
    dim_player: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if dim_player is None:
        dim_player = pd.DataFrame(
            {
                "player_id": fact_player_season["player_id"].drop_duplicates(),
            }
        )
        dim_player["name_full"] = dim_player["player_id"].map(lambda p: f"Player {p}")
        dim_player["name_first"] = "Player"
        dim_player["name_last"] = dim_player["player_id"]

    con = duckdb.connect(":memory:")
    try:
        con.register("fact_player_df", fact_player_season)
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_df")
        con.register("dim_team_df", dim_team)
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")
        con.register("dim_player_df", dim_player)
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _base_player_row(**overrides) -> dict:
    row = {
        "player_id": "player-1",
        "season_key": 2020,
        "team_id": "NYA",
        "player_type": "batter",
        "pa": 0,
        "hr": 0,
        "bb": 0,
        "woba": None,
        "batting_war": 0.0,
        "ip": 0.0,
        "fip": None,
        "era": None,
        "pitching_war": 0.0,
        "player_war": 0.0,
        "salary": 0.0,
        "surplus_value": 0.0,
        "contract_label": "fair_value",
    }
    row.update(overrides)
    return row


def test_player_query_consolidates_traded_player_stints_with_weighted_batting_rate() -> None:
    fact = pd.DataFrame(
        [
            _base_player_row(
                player_id="traded",
                team_id="NYA",
                pa=100,
                hr=10,
                bb=20,
                woba=0.300,
                batting_war=1.0,
                player_war=1.0,
                salary=1_000_000.0,
                surplus_value=7_000_000.0,
                contract_label="fair_value",
            ),
            _base_player_row(
                player_id="traded",
                team_id="BOS",
                pa=300,
                hr=20,
                bb=30,
                woba=0.400,
                batting_war=3.0,
                player_war=3.0,
                salary=2_000_000.0,
                surplus_value=22_000_000.0,
                contract_label="surplus_value",
            ),
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["NYA_2020", "BOS_2020"],
            "team_id": ["NYA", "BOS"],
            "team_name": ["New York Yankees", "Boston Red Sox"],
        }
    )
    players = pd.DataFrame(
        {
            "player_id": ["traded"],
            "name_full": ["Traded Player"],
            "name_first": ["Traded"],
            "name_last": ["Player"],
        }
    )

    result = _run_player_query(fact, teams, players)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == 400
    assert row["hr"] == 30
    assert row["bb"] == 50
    assert row["woba"] == pytest.approx(0.375)
    assert row["batting_war"] == pytest.approx(4.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000.0)
    assert row["surplus_value"] == pytest.approx(29_000_000.0)
    assert row["contract_label"] == "surplus_value"


def test_player_query_joins_dim_team_by_season_to_avoid_historical_name_fanout() -> None:
    fact = pd.DataFrame(
        [
            _base_player_row(
                player_id="yankee",
                season_key=2020,
                team_id="NYA",
                pa=50,
                hr=4,
                bb=8,
                woba=0.350,
                batting_war=2.0,
                player_war=2.0,
                salary=500_000.0,
                surplus_value=15_500_000.0,
            )
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["NYA_2019", "NYA_2020", "NYA_2021"],
            "team_id": ["NYA", "NYA", "NYA"],
            "team_name": ["Historical Yankees", "New York Yankees", "Future Yankees"],
        }
    )

    result = _run_player_query(fact, teams)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["pa"] == 50
    assert row["player_war"] == pytest.approx(2.0)
    assert row["salary"] == pytest.approx(500_000.0)


def test_player_query_prioritizes_player_type_and_weights_pitching_rates() -> None:
    fact = pd.DataFrame(
        [
            _base_player_row(
                player_id="two-way",
                season_key=2021,
                team_id="ANA",
                player_type="pitcher",
                ip=20.0,
                fip=5.0,
                era=4.0,
                pitching_war=1.0,
                player_war=1.0,
            ),
            _base_player_row(
                player_id="two-way",
                season_key=2021,
                team_id="ANA",
                player_type="both",
                pa=80,
                woba=0.450,
                batting_war=2.0,
                ip=10.0,
                fip=2.0,
                era=1.0,
                pitching_war=1.5,
                player_war=3.5,
                contract_label="surplus_value",
            ),
            _base_player_row(
                player_id="two-way",
                season_key=2021,
                team_id="ANA",
                player_type="batter",
                pa=20,
                woba=0.250,
                batting_war=0.1,
                player_war=0.1,
            ),
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["ANA_2021"],
            "team_id": ["ANA"],
            "team_name": ["Los Angeles Angels"],
        }
    )

    result = _run_player_query(fact, teams)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_type"] == "both"
    assert row["pa"] == 100
    assert row["woba"] == pytest.approx((80 * 0.450 + 20 * 0.250) / 100)
    assert row["ip"] == pytest.approx(30.0)
    assert row["fip"] == pytest.approx((20 * 5.0 + 10 * 2.0) / 30)
    assert row["era"] == pytest.approx((20 * 4.0 + 10 * 1.0) / 30)
    assert row["player_war"] == pytest.approx(4.6)
    assert row["contract_label"] == "surplus_value"
