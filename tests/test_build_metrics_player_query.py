from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    db = duckdb.connect(":memory:")
    db.execute(WAREHOUSE_DDL)
    try:
        yield db
    finally:
        db.close()


def _insert_rows(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register("rows_df", df)
    try:
        cols = ", ".join(df.columns)
        con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM rows_df")
    finally:
        con.unregister("rows_df")


def test_player_query_aggregates_traded_player_without_team_name_fanout(
    con: duckdb.DuckDBPyConnection,
) -> None:
    _insert_rows(
        con,
        "dim_team",
        [
            {
                "team_key": "NYA_1990",
                "team_id": "NYA",
                "franchise_id": "NYY",
                "team_name": "New York Highlanders",
                "league_id": "AL",
            },
            {
                "team_key": "NYA_2024",
                "team_id": "NYA",
                "franchise_id": "NYY",
                "team_name": "New York Yankees",
                "league_id": "AL",
            },
            {
                "team_key": "BOS_2024",
                "team_id": "BOS",
                "franchise_id": "BOS",
                "team_name": "Boston Red Sox",
                "league_id": "AL",
            },
        ],
    )
    _insert_rows(
        con,
        "dim_player",
        [
            {
                "player_id": "traded",
                "name_first": "Trade",
                "name_last": "Candidate",
                "name_full": "Trade Candidate",
                "birth_year": 1995,
                "birth_country": "USA",
                "throws": "R",
                "bats": "L",
            }
        ],
    )
    _insert_rows(
        con,
        "fact_player_season",
        [
            {
                "player_id": "traded",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100.0,
                "hr": 5.0,
                "bb": 10.0,
                "woba": 0.400,
                "batting_war": 1.0,
                "ip": 10.0,
                "fip": 2.00,
                "era": 3.00,
                "pitching_war": 0.2,
                "player_war": 1.2,
                "salary": 1_000_000.0,
                "surplus_value": 5_000_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "traded",
                "season_key": 2024,
                "team_id": "BOS",
                "player_type": "pitcher",
                "pa": 300.0,
                "hr": 15.0,
                "bb": 40.0,
                "woba": 0.300,
                "batting_war": 2.0,
                "ip": 30.0,
                "fip": 5.00,
                "era": 6.00,
                "pitching_war": 0.8,
                "player_war": 2.8,
                "salary": 2_000_000.0,
                "surplus_value": 8_000_000.0,
                "contract_label": "fair_value",
            },
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "traded"
    assert row["year_id"] == 2024
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == pytest.approx(400.0)
    assert row["hr"] == pytest.approx(20.0)
    assert row["bb"] == pytest.approx(50.0)
    assert row["woba"] == pytest.approx(0.325)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(4.25)
    assert row["era"] == pytest.approx(5.25)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000.0)
    assert row["surplus_value"] == pytest.approx(13_000_000.0)
    assert row["contract_label"] == "fair_value"


def test_player_query_keeps_same_name_players_distinct(
    con: duckdb.DuckDBPyConnection,
) -> None:
    _insert_rows(
        con,
        "dim_team",
        [
            {
                "team_key": "NYA_2024",
                "team_id": "NYA",
                "franchise_id": "NYY",
                "team_name": "New York Yankees",
                "league_id": "AL",
            }
        ],
    )
    _insert_rows(
        con,
        "dim_player",
        [
            {
                "player_id": "smith01",
                "name_first": "Alex",
                "name_last": "Smith",
                "name_full": "Alex Smith",
                "birth_year": 1990,
                "birth_country": "USA",
                "throws": "R",
                "bats": "R",
            },
            {
                "player_id": "smith02",
                "name_first": "Alex",
                "name_last": "Smith",
                "name_full": "Alex Smith",
                "birth_year": 1998,
                "birth_country": "USA",
                "throws": "L",
                "bats": "L",
            },
        ],
    )
    _insert_rows(
        con,
        "fact_player_season",
        [
            {
                "player_id": "smith01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 200.0,
                "hr": 7.0,
                "bb": 20.0,
                "woba": 0.310,
                "batting_war": 1.5,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "salary": 750_000.0,
                "surplus_value": 6_000_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "smith02",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 50.0,
                "hr": 1.0,
                "bb": 5.0,
                "woba": 0.280,
                "batting_war": 0.1,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.1,
                "salary": 720_000.0,
                "surplus_value": 200_000.0,
                "contract_label": "fair_value",
            },
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert set(result["player_id"]) == {"smith01", "smith02"}
    assert result["name_full"].tolist() == ["Alex Smith", "Alex Smith"]
    assert result.set_index("player_id").loc["smith01", "pa"] == pytest.approx(200.0)
    assert result.set_index("player_id").loc["smith02", "pa"] == pytest.approx(50.0)
