from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _query_players(
    fact_rows: list[dict],
    player_rows: list[dict],
    team_rows: list[dict],
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.register("fact_player_season_df", pd.DataFrame(fact_rows))
        con.register("dim_player_df", pd.DataFrame(player_rows))
        con.register("dim_team_df", pd.DataFrame(team_rows))
        con.execute("CREATE TABLE fact_player_season AS SELECT * FROM fact_player_season_df")
        con.execute("CREATE TABLE dim_player AS SELECT * FROM dim_player_df")
        con.execute("CREATE TABLE dim_team AS SELECT * FROM dim_team_df")
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _base_fact_row(**overrides) -> dict:
    row = {
        "player_id": "player_a",
        "season_key": 2024,
        "team_id": "NYA",
        "player_type": "batter",
        "pa": 0.0,
        "hr": 0.0,
        "bb": 0.0,
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


def test_player_query_collapses_traded_player_with_weighted_rates() -> None:
    result = _query_players(
        fact_rows=[
            _base_fact_row(
                team_id="OAK",
                player_type="batter",
                pa=50,
                hr=2,
                bb=5,
                woba=0.200,
                batting_war=0.4,
                ip=20,
                fip=3.00,
                era=2.00,
                pitching_war=0.6,
                player_war=1.0,
                salary=1_000_000,
                surplus_value=2_000_000,
                contract_label="surplus_value",
            ),
            _base_fact_row(
                team_id="NYA",
                player_type="pitcher",
                pa=150,
                hr=8,
                bb=20,
                woba=0.400,
                batting_war=1.1,
                ip=80,
                fip=5.00,
                era=4.00,
                pitching_war=0.9,
                player_war=2.0,
                salary=4_000_000,
                surplus_value=3_000_000,
                contract_label="fair_value",
            ),
        ],
        player_rows=[
            {
                "player_id": "player_a",
                "name_full": "Two Team Star",
                "name_first": "Two",
                "name_last": "Star",
            }
        ],
        team_rows=[
            {"team_key": "OAK_2024", "team_id": "OAK", "team_name": "Athletics"},
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "Yankees"},
        ],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "Yankees"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 200
    assert row["hr"] == 10
    assert row["bb"] == 25
    assert row["woba"] == pytest.approx(((50 * 0.200) + (150 * 0.400)) / 200)
    assert row["ip"] == 100
    assert row["fip"] == pytest.approx(((20 * 3.00) + (80 * 5.00)) / 100)
    assert row["era"] == pytest.approx(((20 * 2.00) + (80 * 4.00)) / 100)
    assert row["player_war"] == pytest.approx(3.0)
    assert row["salary"] == pytest.approx(5_000_000)
    assert row["surplus_value"] == pytest.approx(5_000_000)
    assert row["contract_label"] == "fair_value"


def test_player_query_joins_dim_team_by_season_without_fanout() -> None:
    result = _query_players(
        fact_rows=[
            _base_fact_row(
                player_id="player_a",
                season_key=2024,
                team_id="NYA",
                pa=100,
                hr=12,
                bb=10,
                woba=0.350,
                batting_war=2.0,
                player_war=2.0,
                salary=6_000_000,
                surplus_value=8_000_000,
            )
        ],
        player_rows=[
            {
                "player_id": "player_a",
                "name_full": "Season Keyed",
                "name_first": "Season",
                "name_last": "Keyed",
            }
        ],
        team_rows=[
            {"team_key": "NYA_2023", "team_id": "NYA", "team_name": "Old Yankees"},
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "New York Yankees"},
            {"team_key": "NYA_2025", "team_id": "NYA", "team_name": "Future Yankees"},
        ],
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["pa"] == 100
    assert row["player_war"] == pytest.approx(2.0)
    assert row["salary"] == pytest.approx(6_000_000)


def test_player_query_preserves_same_name_players_by_player_id() -> None:
    result = _query_players(
        fact_rows=[
            _base_fact_row(player_id="smitha01", player_war=3.0, salary=1_000_000),
            _base_fact_row(player_id="smithb01", player_war=1.0, salary=2_000_000),
        ],
        player_rows=[
            {
                "player_id": "smitha01",
                "name_full": "John Smith",
                "name_first": "John",
                "name_last": "Smith",
            },
            {
                "player_id": "smithb01",
                "name_full": "John Smith",
                "name_first": "John",
                "name_last": "Smith",
            },
        ],
        team_rows=[
            {"team_key": "NYA_2024", "team_id": "NYA", "team_name": "Yankees"},
        ],
    )

    assert len(result) == 2
    assert set(result["player_id"]) == {"smitha01", "smithb01"}
    assert result.set_index("player_id").loc["smitha01", "salary"] == pytest.approx(1_000_000)
    assert result.set_index("player_id").loc["smithb01", "salary"] == pytest.approx(2_000_000)
