from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def test_player_query_aggregates_traded_player_without_team_history_fanout() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.register(
            "fact_player_season",
            pd.DataFrame(
                [
                    {
                        "player_id": "player-1",
                        "season_key": 2020,
                        "team_id": "AAA",
                        "player_type": "batter",
                        "pa": 100,
                        "hr": 10,
                        "bb": 20,
                        "woba": 0.400,
                        "batting_war": 2.5,
                        "ip": 10.0,
                        "fip": 2.00,
                        "era": 3.00,
                        "pitching_war": 0.5,
                        "player_war": 3.0,
                        "salary": 10_000_000,
                        "surplus_value": 14_000_000,
                        "contract_label": "surplus_value",
                    },
                    {
                        "player_id": "player-1",
                        "season_key": 2020,
                        "team_id": "BBB",
                        "player_type": "pitcher",
                        "pa": 300,
                        "hr": 5,
                        "bb": 30,
                        "woba": 0.300,
                        "batting_war": 0.2,
                        "ip": 30.0,
                        "fip": 4.00,
                        "era": 5.00,
                        "pitching_war": 0.8,
                        "player_war": 1.0,
                        "salary": 5_000_000,
                        "surplus_value": 3_000_000,
                        "contract_label": "fair_value",
                    },
                ]
            ),
        )
        con.register(
            "dim_player",
            pd.DataFrame(
                [
                    {
                        "player_id": "player-1",
                        "name_full": "Trade Target",
                        "name_first": "Trade",
                        "name_last": "Target",
                    }
                ]
            ),
        )
        con.register(
            "dim_team",
            pd.DataFrame(
                [
                    {
                        "team_key": "AAA_2020",
                        "team_id": "AAA",
                        "team_name": "2020 Name",
                    },
                    {
                        "team_key": "AAA_2021",
                        "team_id": "AAA",
                        "team_name": "Renamed Club",
                    },
                    {
                        "team_key": "BBB_2020",
                        "team_id": "BBB",
                        "team_name": "Other Club",
                    },
                ]
            ),
        )

        result = con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-1"
    assert row["team_id"] == "AAA"
    assert row["team_name"] == "2020 Name"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == 400
    assert row["hr"] == 15
    assert row["bb"] == 50
    assert row["woba"] == pytest.approx(0.325)
    assert row["ip"] == pytest.approx(40.0)
    assert row["fip"] == pytest.approx(3.5)
    assert row["era"] == pytest.approx(4.5)
    assert row["batting_war"] == pytest.approx(2.7)
    assert row["pitching_war"] == pytest.approx(1.3)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == 15_000_000
    assert row["surplus_value"] == 17_000_000
    assert row["contract_label"] == "surplus_value"
