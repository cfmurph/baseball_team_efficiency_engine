from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_table(con: duckdb.DuckDBPyConnection, name: str, df: pd.DataFrame) -> None:
    view_name = f"{name}_df"
    con.register(view_name, df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM {view_name}")
    con.unregister(view_name)


def _run_player_query(
    fact_player_season: pd.DataFrame,
    dim_player: pd.DataFrame,
    dim_team: pd.DataFrame,
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        _create_table(con, "fact_player_season", fact_player_season)
        _create_table(con, "dim_player", dim_player)
        _create_table(con, "dim_team", dim_team)
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _fact_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "trade01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100.0,
                "hr": 10.0,
                "bb": 20.0,
                "woba": 0.300,
                "batting_war": 2.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.0,
                "salary": 1_000_000.0,
                "surplus_value": 5_000_000.0,
                "contract_label": "surplus_value",
            },
            {
                "player_id": "trade01",
                "season_key": 2024,
                "team_id": "LAN",
                "player_type": "batter",
                "pa": 300.0,
                "hr": 20.0,
                "bb": 30.0,
                "woba": 0.400,
                "batting_war": 1.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.0,
                "salary": 2_000_000.0,
                "surplus_value": 3_000_000.0,
                "contract_label": "fair_value",
            },
            {
                "player_id": "pitch01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "pitcher",
                "pa": 0.0,
                "hr": 0.0,
                "bb": 0.0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 20.0,
                "fip": 3.0,
                "era": 2.0,
                "pitching_war": 0.5,
                "player_war": 0.5,
                "salary": 500_000.0,
                "surplus_value": 1_000_000.0,
                "contract_label": "rookie_scale",
            },
            {
                "player_id": "pitch01",
                "season_key": 2024,
                "team_id": "LAN",
                "player_type": "pitcher",
                "pa": 0.0,
                "hr": 0.0,
                "bb": 0.0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 80.0,
                "fip": 5.0,
                "era": 4.0,
                "pitching_war": 2.5,
                "player_war": 2.5,
                "salary": 1_500_000.0,
                "surplus_value": 2_000_000.0,
                "contract_label": "fair_value",
            },
            {
                "player_id": "same01",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 50.0,
                "hr": 1.0,
                "bb": 5.0,
                "woba": 0.250,
                "batting_war": 0.2,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.2,
                "salary": 750_000.0,
                "surplus_value": 250_000.0,
                "contract_label": "fair_value",
            },
            {
                "player_id": "same02",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 60.0,
                "hr": 2.0,
                "bb": 6.0,
                "woba": 0.275,
                "batting_war": 0.3,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.3,
                "salary": 800_000.0,
                "surplus_value": 300_000.0,
                "contract_label": "fair_value",
            },
        ]
    )


def _dim_player() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": "trade01", "name_full": "Taylor Traded", "name_first": "Taylor", "name_last": "Traded"},
            {"player_id": "pitch01", "name_full": "Pat Pitcher", "name_first": "Pat", "name_last": "Pitcher"},
            {"player_id": "same01", "name_full": "Chris Young", "name_first": "Chris", "name_last": "Young"},
            {"player_id": "same02", "name_full": "Chris Young", "name_first": "Chris", "name_last": "Young"},
        ]
    )


def _dim_team() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team_key": "NYA_2023",
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
                "team_key": "LAN_2024",
                "team_id": "LAN",
                "franchise_id": "LAD",
                "team_name": "Los Angeles Dodgers",
                "league_id": "NL",
            },
        ]
    )


@pytest.fixture
def player_metrics() -> pd.DataFrame:
    return _run_player_query(_fact_rows(), _dim_player(), _dim_team())


def test_player_query_aggregates_traded_player_to_one_weighted_season(player_metrics: pd.DataFrame) -> None:
    traded = player_metrics[player_metrics["player_id"] == "trade01"]

    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "New York Yankees"
    assert row["player_type"] == "batter"
    assert row["pa"] == 400
    assert row["hr"] == 30
    assert row["bb"] == 50
    assert row["woba"] == pytest.approx(0.375)
    assert row["batting_war"] == 3
    assert row["player_war"] == 3
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 8_000_000
    assert row["contract_label"] == "surplus_value"


def test_player_query_weights_pitching_rates_by_innings_pitched(player_metrics: pd.DataFrame) -> None:
    pitcher = player_metrics[player_metrics["player_id"] == "pitch01"].iloc[0]

    assert pitcher["team_id"] == "LAN"
    assert pitcher["team_name"] == "Los Angeles Dodgers"
    assert pitcher["player_type"] == "pitcher"
    assert pitcher["ip"] == 100
    assert pitcher["fip"] == pytest.approx(4.6)
    assert pitcher["era"] == pytest.approx(3.6)
    assert pitcher["pitching_war"] == 3
    assert pitcher["player_war"] == 3


def test_player_query_uses_season_team_key_without_historical_name_fanout(player_metrics: pd.DataFrame) -> None:
    traded = player_metrics[player_metrics["player_id"] == "trade01"].iloc[0]

    assert traded["pa"] == 400
    assert traded["player_war"] == 3
    assert traded["team_name"] == "New York Yankees"
    assert "New York Highlanders" not in set(player_metrics["team_name"].dropna())


def test_player_query_preserves_distinct_same_name_players(player_metrics: pd.DataFrame) -> None:
    same_name = player_metrics[player_metrics["name_full"] == "Chris Young"]

    assert set(same_name["player_id"]) == {"same01", "same02"}
    assert len(same_name) == 2
