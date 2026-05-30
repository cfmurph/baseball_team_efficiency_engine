from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_table(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register("_test_df", df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM _test_df")
    con.unregister("_test_df")


@pytest.fixture
def player_metrics_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    _create_table(
        con,
        "dim_player",
        [
            {"player_id": "traded-1", "name_full": "Taylor Traded", "name_first": "Taylor", "name_last": "Traded"},
            {"player_id": "smith-a", "name_full": "Chris Smith", "name_first": "Chris", "name_last": "Smith"},
            {"player_id": "smith-b", "name_full": "Chris Smith", "name_first": "Chris", "name_last": "Smith"},
        ],
    )
    _create_table(
        con,
        "dim_team",
        [
            {"team_key": "AAA_2023", "team_id": "AAA", "team_name": "Old Alpha"},
            {"team_key": "AAA_2024", "team_id": "AAA", "team_name": "Alpha"},
            {"team_key": "BBB_2024", "team_id": "BBB", "team_name": "Beta"},
            {"team_key": "CCC_2024", "team_id": "CCC", "team_name": "Gamma"},
        ],
    )
    _create_table(
        con,
        "fact_player_season",
        [
            {
                "player_id": "traded-1",
                "season_key": 2024,
                "team_id": "AAA",
                "player_type": "both",
                "pa": 100,
                "hr": 10,
                "bb": 20,
                "woba": 0.300,
                "batting_war": 1.0,
                "ip": 100.0,
                "fip": 2.0,
                "era": 6.0,
                "pitching_war": 1.0,
                "player_war": 2.0,
                "salary": 1_000_000,
                "surplus_value": 5_000_000,
                "contract_label": "value",
            },
            {
                "player_id": "traded-1",
                "season_key": 2024,
                "team_id": "BBB",
                "player_type": "pitcher",
                "pa": 50,
                "hr": 5,
                "bb": 10,
                "woba": 0.500,
                "batting_war": 0.5,
                "ip": 50.0,
                "fip": 4.0,
                "era": 3.0,
                "pitching_war": 2.0,
                "player_war": 2.5,
                "salary": 2_000_000,
                "surplus_value": 8_000_000,
                "contract_label": "elite",
            },
            {
                "player_id": "smith-a",
                "season_key": 2024,
                "team_id": "CCC",
                "player_type": "batter",
                "pa": 200,
                "hr": 12,
                "bb": 25,
                "woba": 0.330,
                "batting_war": 1.2,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 1.2,
                "salary": 900_000,
                "surplus_value": 4_000_000,
                "contract_label": "value",
            },
            {
                "player_id": "smith-b",
                "season_key": 2024,
                "team_id": "CCC",
                "player_type": "batter",
                "pa": 75,
                "hr": 2,
                "bb": 6,
                "woba": 0.280,
                "batting_war": 0.1,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.1,
                "salary": 600_000,
                "surplus_value": -100_000,
                "contract_label": "dead_money",
            },
        ],
    )
    try:
        yield con
    finally:
        con.close()


def test_player_query_aggregates_traded_player_without_historical_team_fanout(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    row = result[result["player_id"] == "traded-1"].iloc[0]

    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta"
    assert row["player_type"] == "both"
    assert row["pa"] == 150
    assert row["hr"] == 15
    assert row["bb"] == 30
    assert row["batting_war"] == pytest.approx(1.5)
    assert row["ip"] == pytest.approx(150.0)
    assert row["pitching_war"] == pytest.approx(3.0)
    assert row["player_war"] == pytest.approx(4.5)
    assert row["salary"] == 3_000_000
    assert row["surplus_value"] == 13_000_000
    assert row["contract_label"] == "elite"


def test_player_query_weights_rate_stats_by_playing_time(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    row = result[result["player_id"] == "traded-1"].iloc[0]

    assert row["woba"] == pytest.approx(((0.300 * 100) + (0.500 * 50)) / 150)
    assert row["fip"] == pytest.approx(((2.0 * 100) + (4.0 * 50)) / 150)
    assert row["era"] == pytest.approx(((6.0 * 100) + (3.0 * 50)) / 150)


def test_player_query_preserves_distinct_players_with_same_name(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    same_name = result[result["name_full"] == "Chris Smith"]

    assert set(same_name["player_id"]) == {"smith-a", "smith-b"}
    assert len(same_name) == 2
