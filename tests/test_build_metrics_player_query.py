from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


_FACT_COLUMNS = [
    "player_id",
    "season_key",
    "team_id",
    "player_type",
    "pa",
    "hr",
    "bb",
    "woba",
    "batting_war",
    "ip",
    "fip",
    "era",
    "pitching_war",
    "player_war",
    "salary",
    "surplus_value",
    "contract_label",
]

_DIM_PLAYER_COLUMNS = ["player_id", "name_full", "name_first", "name_last"]
_DIM_TEAM_COLUMNS = ["team_key", "team_id", "team_name"]


def _fetch_player_metrics(
    fact_rows: list[dict],
    dim_player_rows: list[dict],
    dim_team_rows: list[dict],
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        fact_df = pd.DataFrame(fact_rows, columns=_FACT_COLUMNS)
        player_df = pd.DataFrame(dim_player_rows, columns=_DIM_PLAYER_COLUMNS)
        team_df = pd.DataFrame(dim_team_rows, columns=_DIM_TEAM_COLUMNS)

        con.register("fact_player_df", fact_df)
        con.register("dim_player_df", player_df)
        con.register("dim_team_df", team_df)
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")

        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def test_player_query_aggregates_traded_player_with_weighted_rates() -> None:
    result = _fetch_player_metrics(
        fact_rows=[
            {
                "player_id": "traded01",
                "season_key": 2020,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100,
                "hr": 10,
                "bb": 20,
                "woba": 0.400,
                "batting_war": 1.0,
                "ip": 10,
                "fip": 4.50,
                "era": 5.00,
                "pitching_war": 0.1,
                "player_war": 1.1,
                "salary": 1_000_000,
                "surplus_value": 5_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "traded01",
                "season_key": 2020,
                "team_id": "BOS",
                "player_type": "both",
                "pa": 300,
                "hr": 20,
                "bb": 30,
                "woba": 0.300,
                "batting_war": 2.0,
                "ip": 30,
                "fip": 3.50,
                "era": 3.00,
                "pitching_war": 0.4,
                "player_war": 2.4,
                "salary": 3_000_000,
                "surplus_value": 7_000_000,
                "contract_label": "fair_value",
            },
        ],
        dim_player_rows=[
            {
                "player_id": "traded01",
                "name_full": "Taylor Traded",
                "name_first": "Taylor",
                "name_last": "Traded",
            }
        ],
        dim_team_rows=[
            {"team_key": "NYA_2020", "team_id": "NYA", "team_name": "Yankees"},
            {"team_key": "BOS_2020", "team_id": "BOS", "team_name": "Red Sox"},
        ],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded01"
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Red Sox"
    assert row["player_type"] == "both"
    assert row["pa"] == 400
    assert row["hr"] == 30
    assert row["bb"] == 50
    assert row["woba"] == pytest.approx((0.400 * 100 + 0.300 * 300) / 400)
    assert row["batting_war"] == pytest.approx(3.0)
    assert row["ip"] == 40
    assert row["fip"] == pytest.approx((4.50 * 10 + 3.50 * 30) / 40)
    assert row["era"] == pytest.approx((5.00 * 10 + 3.00 * 30) / 40)
    assert row["pitching_war"] == pytest.approx(0.5)
    assert row["player_war"] == pytest.approx(3.5)
    assert row["salary"] == 4_000_000
    assert row["surplus_value"] == 12_000_000
    assert row["contract_label"] == "fair_value"


def test_player_query_joins_dim_team_by_season_without_fanout() -> None:
    result = _fetch_player_metrics(
        fact_rows=[
            {
                "player_id": "steady01",
                "season_key": 2020,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 50,
                "hr": 5,
                "bb": 8,
                "woba": 0.250,
                "batting_war": 0.8,
                "ip": 0,
                "fip": None,
                "era": None,
                "pitching_war": 0,
                "player_war": 0.8,
                "salary": 750_000,
                "surplus_value": 1_500_000,
                "contract_label": "fair_value",
            }
        ],
        dim_player_rows=[
            {
                "player_id": "steady01",
                "name_full": "Sam Steady",
                "name_first": "Sam",
                "name_last": "Steady",
            }
        ],
        dim_team_rows=[
            {"team_key": "NYA_2019", "team_id": "NYA", "team_name": "Old Yankees"},
            {"team_key": "NYA_2020", "team_id": "NYA", "team_name": "Yankees"},
            {"team_key": "NYA_2021", "team_id": "NYA", "team_name": "Future Yankees"},
        ],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Yankees"
    assert row["pa"] == 50
    assert row["hr"] == 5
    assert row["player_war"] == pytest.approx(0.8)


def test_player_query_keeps_same_name_players_separate_by_player_id() -> None:
    result = _fetch_player_metrics(
        fact_rows=[
            {
                "player_id": "smith01",
                "season_key": 2021,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100,
                "hr": 7,
                "bb": 12,
                "woba": 0.310,
                "batting_war": 1.2,
                "ip": 0,
                "fip": None,
                "era": None,
                "pitching_war": 0,
                "player_war": 1.2,
                "salary": 1_000_000,
                "surplus_value": 2_000_000,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "smith02",
                "season_key": 2021,
                "team_id": "BOS",
                "player_type": "pitcher",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": None,
                "batting_war": 0,
                "ip": 80,
                "fip": 3.25,
                "era": 3.10,
                "pitching_war": 2.4,
                "player_war": 2.4,
                "salary": 2_000_000,
                "surplus_value": 4_000_000,
                "contract_label": "surplus_value",
            },
        ],
        dim_player_rows=[
            {
                "player_id": "smith01",
                "name_full": "Alex Smith",
                "name_first": "Alex",
                "name_last": "Smith",
            },
            {
                "player_id": "smith02",
                "name_full": "Alex Smith",
                "name_first": "Alex",
                "name_last": "Smith",
            },
        ],
        dim_team_rows=[
            {"team_key": "NYA_2021", "team_id": "NYA", "team_name": "Yankees"},
            {"team_key": "BOS_2021", "team_id": "BOS", "team_name": "Red Sox"},
        ],
    )

    assert set(result["player_id"]) == {"smith01", "smith02"}
    assert len(result) == 2
    assert result.groupby("name_full")["player_id"].nunique().to_dict() == {"Alex Smith": 2}
