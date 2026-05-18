from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _run_player_query(
    fact_rows: list[dict],
    dim_team_rows: list[dict],
    dim_player_rows: list[dict],
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        con.register("fact_player_season", pd.DataFrame(fact_rows))
        con.register("dim_team", pd.DataFrame(dim_team_rows))
        con.register("dim_player", pd.DataFrame(dim_player_rows))
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _player_row(player_id: str, **overrides) -> dict:
    row = {
        "player_id": player_id,
        "season_key": 2024,
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
        "salary": 0,
        "surplus_value": 0,
        "contract_label": "fair_value",
    }
    row.update(overrides)
    return row


def _dim_player(player_id: str, name_full: str) -> dict:
    name_first, name_last = name_full.split(" ", 1)
    return {
        "player_id": player_id,
        "name_full": name_full,
        "name_first": name_first,
        "name_last": name_last,
    }


DIM_TEAMS_2024 = [
    {
        "team_key": "OAK_2024",
        "team_id": "OAK",
        "team_name": "Oakland Athletics",
    },
    {
        "team_key": "NYA_2024",
        "team_id": "NYA",
        "team_name": "New York Yankees",
    },
    {
        "team_key": "BOS_2024",
        "team_id": "BOS",
        "team_name": "Boston Red Sox",
    },
]


def test_player_query_collapses_traded_stints_with_weighted_rates():
    df = _run_player_query(
        [
            _player_row(
                "traded01",
                team_id="OAK",
                pa=100,
                hr=2,
                bb=10,
                woba=0.300,
                batting_war=1.0,
                player_war=1.0,
                salary=1_000_000,
                surplus_value=2_000_000,
                contract_label="fair_value",
            ),
            _player_row(
                "traded01",
                team_id="NYA",
                pa=300,
                hr=8,
                bb=30,
                woba=0.400,
                batting_war=4.0,
                player_war=4.0,
                salary=2_000_000,
                surplus_value=8_000_000,
                contract_label="surplus_value",
            ),
            _player_row(
                "pitch01",
                team_id="BOS",
                player_type="pitcher",
                ip=50.0,
                fip=2.00,
                era=4.00,
                pitching_war=1.0,
                player_war=1.0,
            ),
            _player_row(
                "pitch01",
                team_id="NYA",
                player_type="pitcher",
                ip=150.0,
                fip=4.00,
                era=2.00,
                pitching_war=3.0,
                player_war=3.0,
            ),
        ],
        DIM_TEAMS_2024,
        [
            _dim_player("traded01", "Traded Batter"),
            _dim_player("pitch01", "Traded Pitcher"),
        ],
    )

    assert set(df["player_id"]) == {"traded01", "pitch01"}

    batter = df[df["player_id"] == "traded01"].iloc[0]
    assert batter["team_id"] == "NYA"
    assert batter["team_name"] == "New York Yankees"
    assert batter["pa"] == 400
    assert batter["hr"] == 10
    assert batter["bb"] == 40
    assert batter["woba"] == pytest.approx(0.375)
    assert batter["batting_war"] == pytest.approx(5.0)
    assert batter["salary"] == 3_000_000
    assert batter["surplus_value"] == 10_000_000
    assert batter["contract_label"] == "surplus_value"

    pitcher = df[df["player_id"] == "pitch01"].iloc[0]
    assert pitcher["team_id"] == "NYA"
    assert pitcher["ip"] == pytest.approx(200.0)
    assert pitcher["fip"] == pytest.approx(3.5)
    assert pitcher["era"] == pytest.approx(2.5)
    assert pitcher["pitching_war"] == pytest.approx(4.0)


def test_player_query_joins_dim_team_by_season_specific_team_key():
    df = _run_player_query(
        [
            _player_row(
                "rename01",
                team_id="ANA",
                pa=25,
                hr=3,
                woba=0.500,
                batting_war=1.2,
                player_war=1.2,
            ),
        ],
        [
            {
                "team_key": "ANA_2023",
                "team_id": "ANA",
                "team_name": "Anaheim Angels",
            },
            {
                "team_key": "ANA_2024",
                "team_id": "ANA",
                "team_name": "Los Angeles Angels",
            },
        ],
        [_dim_player("rename01", "Renamed Team")],
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["team_name"] == "Los Angeles Angels"
    assert row["pa"] == 25
    assert row["player_war"] == pytest.approx(1.2)


def test_player_query_preserves_same_name_players_as_distinct_people():
    df = _run_player_query(
        [
            _player_row("gonzale01", pa=200, player_war=2.0),
            _player_row("gonzale02", pa=100, player_war=1.0),
        ],
        DIM_TEAMS_2024,
        [
            _dim_player("gonzale01", "Alex Gonzalez"),
            _dim_player("gonzale02", "Alex Gonzalez"),
        ],
    )

    assert len(df) == 2
    assert set(df["player_id"]) == {"gonzale01", "gonzale02"}
    assert df["name_full"].tolist() == ["Alex Gonzalez", "Alex Gonzalez"]
