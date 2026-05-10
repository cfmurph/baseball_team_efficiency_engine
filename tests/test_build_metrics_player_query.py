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
        con.register("fact_player_season", fact_player_season)
        con.register("dim_player", dim_player)
        con.register("dim_team", dim_team)
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _player_dim(*player_ids: str, name_full: str = "Shared Name") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": list(player_ids),
            "name_full": [name_full] * len(player_ids),
            "name_first": [name_full.split()[0]] * len(player_ids),
            "name_last": [name_full.split()[-1]] * len(player_ids),
        }
    )


def test_player_query_consolidates_traded_player_with_weighted_rates() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "traded1",
                "season_key": 2024,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 100,
                "hr": 5,
                "bb": 10,
                "woba": 0.300,
                "batting_war": 1.0,
                "ip": 10.0,
                "fip": 4.00,
                "era": 3.00,
                "pitching_war": 0.2,
                "player_war": 1.2,
                "salary": 1_000_000,
                "surplus_value": 5_000_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "traded1",
                "season_key": 2024,
                "team_id": "BBB",
                "player_type": "pitcher",
                "pa": 300,
                "hr": 20,
                "bb": 30,
                "woba": 0.400,
                "batting_war": 3.0,
                "ip": 30.0,
                "fip": 2.00,
                "era": 5.00,
                "pitching_war": 0.8,
                "player_war": 3.8,
                "salary": 2_000_000,
                "surplus_value": 20_000_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["AAA_2024", "BBB_2024"],
            "team_id": ["AAA", "BBB"],
            "team_name": ["Alpha Aces", "Beta Bears"],
        }
    )

    result = _run_player_query(fact, _player_dim("traded1", name_full="Trade Target"), teams)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Bears"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 400
    assert row["hr"] == 25
    assert row["bb"] == 40
    assert row["salary"] == 3_000_000
    assert row["player_war"] == pytest.approx(5.0)
    assert row["woba"] == pytest.approx(0.375)
    assert row["fip"] == pytest.approx(2.5)
    assert row["era"] == pytest.approx(4.5)
    assert row["contract_label"] == "surplus_value"


def test_player_query_joins_dim_team_by_season_to_prevent_fanout() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "steady1",
                "season_key": 2024,
                "team_id": "NYA",
                "player_type": "batter",
                "pa": 100,
                "hr": 12,
                "bb": 14,
                "woba": 0.350,
                "batting_war": 2.0,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 2.0,
                "salary": 750_000,
                "surplus_value": 9_250_000,
                "contract_label": "surplus_value",
            }
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["NYA_2023", "NYA_2024"],
            "team_id": ["NYA", "NYA"],
            "team_name": ["Old New York Name", "New York Yankees"],
        }
    )

    result = _run_player_query(fact, _player_dim("steady1", name_full="Steady Hitter"), teams)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "New York Yankees"
    assert row["pa"] == 100
    assert row["hr"] == 12
    assert row["salary"] == 750_000
    assert row["player_war"] == pytest.approx(2.0)


def test_player_query_preserves_distinct_people_who_share_a_name() -> None:
    fact = pd.DataFrame(
        [
            {
                "player_id": "smith-a",
                "season_key": 2024,
                "team_id": "AAA",
                "player_type": "batter",
                "pa": 50,
                "hr": 1,
                "bb": 5,
                "woba": 0.310,
                "batting_war": 0.4,
                "ip": 0.0,
                "fip": None,
                "era": None,
                "pitching_war": 0.0,
                "player_war": 0.4,
                "salary": 500_000,
                "surplus_value": 1_500_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "smith-b",
                "season_key": 2024,
                "team_id": "BBB",
                "player_type": "pitcher",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": None,
                "batting_war": 0.0,
                "ip": 25.0,
                "fip": 3.20,
                "era": 3.60,
                "pitching_war": 0.9,
                "player_war": 0.9,
                "salary": 600_000,
                "surplus_value": 3_900_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    teams = pd.DataFrame(
        {
            "team_key": ["AAA_2024", "BBB_2024"],
            "team_id": ["AAA", "BBB"],
            "team_name": ["Alpha Aces", "Beta Bears"],
        }
    )

    result = _run_player_query(fact, _player_dim("smith-a", "smith-b", name_full="Chris Smith"), teams)

    assert len(result) == 2
    assert set(result["player_id"]) == {"smith-a", "smith-b"}
    assert result["name_full"].tolist() == ["Chris Smith", "Chris Smith"]
