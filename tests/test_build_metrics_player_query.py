from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _run_player_query(
    fact_player_season: pd.DataFrame,
    dim_player: pd.DataFrame,
    dim_team: pd.DataFrame,
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.register("fact_player_season_df", fact_player_season)
        con.register("dim_player_df", dim_player)
        con.register("dim_team_df", dim_team)
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _base_dim_player() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "traded",
                "name_full": "Traded Player",
                "name_first": "Traded",
                "name_last": "Player",
            },
            {
                "player_id": "same_name_a",
                "name_full": "Alex Gonzalez",
                "name_first": "Alex",
                "name_last": "Gonzalez",
            },
            {
                "player_id": "same_name_b",
                "name_full": "Alex Gonzalez",
                "name_first": "Alex",
                "name_last": "Gonzalez",
            },
        ]
    )


def _base_dim_team() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_key": "NYA_2020",
                "team_id": "NYA",
                "team_name": "New York Yankees",
            },
            {
                "team_key": "BOS_2020",
                "team_id": "BOS",
                "team_name": "Boston Red Sox",
            },
        ]
    )


def test_player_query_aggregates_traded_player_with_playing_time_weighted_rates() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "traded",
                "season_key": 2020,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 10.0,
                "hr": 1.0,
                "bb": 2.0,
                "woba": 0.500,
                "batting_war": 0.5,
                "ip": 10.0,
                "fip": 5.00,
                "era": 6.00,
                "pitching_war": 0.1,
                "player_war": 0.6,
                "salary": 1_000_000.0,
                "surplus_value": 4_000_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "traded",
                "season_key": 2020,
                "team_id": "BOS",
                "player_type": "batter",
                "pa": 90.0,
                "hr": 9.0,
                "bb": 18.0,
                "woba": 0.300,
                "batting_war": 1.5,
                "ip": 90.0,
                "fip": 3.00,
                "era": 4.00,
                "pitching_war": 0.9,
                "player_war": 2.4,
                "salary": 2_000_000.0,
                "surplus_value": 16_000_000.0,
                "contract_label": "fair_value",
            },
        ]
    )

    result = _run_player_query(fact, _base_dim_player(), _base_dim_team())

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded"
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["pa"] == pytest.approx(100.0)
    assert row["hr"] == pytest.approx(10.0)
    assert row["bb"] == pytest.approx(20.0)
    assert row["woba"] == pytest.approx((10.0 * 0.500 + 90.0 * 0.300) / 100.0)
    assert row["ip"] == pytest.approx(100.0)
    assert row["fip"] == pytest.approx((10.0 * 5.00 + 90.0 * 3.00) / 100.0)
    assert row["era"] == pytest.approx((10.0 * 6.00 + 90.0 * 4.00) / 100.0)
    assert row["batting_war"] == pytest.approx(2.0)
    assert row["pitching_war"] == pytest.approx(1.0)
    assert row["player_war"] == pytest.approx(3.0)
    assert row["salary"] == pytest.approx(3_000_000.0)
    assert row["surplus_value"] == pytest.approx(20_000_000.0)
    assert row["contract_label"] == "fair_value"


def test_player_query_joins_team_by_season_key_without_historical_name_fanout() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "traded",
                "season_key": 2020,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100.0,
                "hr": 10.0,
                "bb": 20.0,
                "woba": 0.350,
                "batting_war": 2.0,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.0,
                "salary": 1_000_000.0,
                "surplus_value": 15_000_000.0,
                "contract_label": "surplus_value",
            }
        ]
    )
    dim_team = pd.DataFrame(
        [
            {
                "team_key": "NYA_2020",
                "team_id": "NYA",
                "team_name": "New York Yankees",
            },
            {
                "team_key": "NYA_2021",
                "team_id": "NYA",
                "team_name": "New York Highlanders",
            },
        ]
    )

    result = _run_player_query(fact, _base_dim_player(), dim_team)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["pa"] == pytest.approx(100.0)
    assert row["player_war"] == pytest.approx(2.0)
    assert row["salary"] == pytest.approx(1_000_000.0)


def test_player_query_preserves_distinct_players_who_share_a_name() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "same_name_a",
                "season_key": 2020,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 250.0,
                "hr": 5.0,
                "bb": 30.0,
                "woba": 0.310,
                "batting_war": 1.2,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.2,
                "salary": 500_000.0,
                "surplus_value": 9_100_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "same_name_b",
                "season_key": 2020,
                "team_id": "BOS",
                "player_type": "batter",
                "pa": 300.0,
                "hr": 7.0,
                "bb": 25.0,
                "woba": 0.330,
                "batting_war": 1.8,
                "ip": None,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.8,
                "salary": 750_000.0,
                "surplus_value": 13_650_000.0,
                "contract_label": "surplus_value",
            },
        ]
    )

    result = _run_player_query(fact, _base_dim_player(), _base_dim_team())

    assert len(result) == 2
    assert result["name_full"].tolist() == ["Alex Gonzalez", "Alex Gonzalez"]
    assert set(result["player_id"]) == {"same_name_a", "same_name_b"}
    assert result.set_index("player_id").loc["same_name_a", "team_name"] == "New York Yankees"
    assert result.set_index("player_id").loc["same_name_b", "team_name"] == "Boston Red Sox"
