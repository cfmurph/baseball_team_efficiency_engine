from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_table(con: duckdb.DuckDBPyConnection, name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    view_name = f"{name}_df"
    con.register(view_name, df)
    con.execute(f"CREATE TABLE {name} AS SELECT * FROM {view_name}")
    con.unregister(view_name)


def _query_player_metrics(
    fact_rows: list[dict],
    player_rows: list[dict],
    team_rows: list[dict],
) -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    try:
        _create_table(con, "fact_player_season", fact_rows)
        _create_table(con, "dim_player", player_rows)
        _create_table(con, "dim_team", team_rows)
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def _fact_row(
    player_id: str,
    team_id: str,
    *,
    player_war: float,
    salary: int,
    season_key: int = 2024,
    player_type: str = "batter",
    pa: int = 100,
    hr: int = 10,
    bb: int = 20,
    woba: float = 0.340,
    batting_war: float = 1.0,
    ip: float = 0.0,
    fip: float | None = None,
    era: float | None = None,
    pitching_war: float = 0.0,
    surplus_value: float = 1_000_000.0,
    contract_label: str = "fair_value",
) -> dict:
    return {
        "player_id": player_id,
        "season_key": season_key,
        "team_id": team_id,
        "player_type": player_type,
        "pa": pa,
        "hr": hr,
        "bb": bb,
        "woba": woba,
        "batting_war": batting_war,
        "ip": ip,
        "fip": fip,
        "era": era,
        "pitching_war": pitching_war,
        "player_war": player_war,
        "salary": salary,
        "surplus_value": surplus_value,
        "contract_label": contract_label,
    }


def test_player_query_keeps_same_name_players_distinct_by_player_id() -> None:
    df = _query_player_metrics(
        fact_rows=[
            _fact_row("smithjo01", "NYA", player_war=2.5, salary=5_000_000),
            _fact_row("smithjo02", "BOS", player_war=1.5, salary=1_500_000),
        ],
        player_rows=[
            {"player_id": "smithjo01", "name_full": "John Smith", "name_first": "John", "name_last": "Smith"},
            {"player_id": "smithjo02", "name_full": "John Smith", "name_first": "John", "name_last": "Smith"},
        ],
        team_rows=[
            {"team_id": "NYA", "team_name": "Yankees"},
            {"team_id": "BOS", "team_name": "Red Sox"},
        ],
    )

    same_name = df[df["name_full"] == "John Smith"].sort_values("player_id").reset_index(drop=True)
    assert same_name["player_id"].tolist() == ["smithjo01", "smithjo02"]
    assert same_name["team_name"].tolist() == ["Yankees", "Red Sox"]
    assert same_name["player_war"].tolist() == [2.5, 1.5]


def test_player_query_collapses_traded_player_stints_to_one_season_row() -> None:
    df = _query_player_metrics(
        fact_rows=[
            _fact_row(
                "traded01",
                "NYA",
                player_war=3.0,
                salary=4_000_000,
                pa=300,
                hr=20,
                bb=40,
                batting_war=2.5,
                surplus_value=20_000_000.0,
                contract_label="surplus_value",
            ),
            _fact_row(
                "traded01",
                "BOS",
                player_war=1.5,
                salary=2_000_000,
                pa=150,
                hr=8,
                bb=15,
                batting_war=1.0,
                surplus_value=10_000_000.0,
                contract_label="fair_value",
            ),
        ],
        player_rows=[
            {"player_id": "traded01", "name_full": "Traded Player", "name_first": "Traded", "name_last": "Player"},
        ],
        team_rows=[
            {"team_id": "NYA", "team_name": "Yankees"},
            {"team_id": "BOS", "team_name": "Red Sox"},
        ],
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_id"] == "traded01"
    assert row["team_id"] == "NYA"
    assert row["team_name"] == "Yankees"
    assert row["contract_label"] == "surplus_value"
    assert row["pa"] == 450
    assert row["hr"] == 28
    assert row["bb"] == 55
    assert row["batting_war"] == 3.5
    assert row["player_war"] == 4.5
    assert row["salary"] == 6_000_000
    assert row["surplus_value"] == 30_000_000.0


def test_player_query_deduplicates_repeated_team_dimension_rows() -> None:
    df = _query_player_metrics(
        fact_rows=[
            _fact_row("judgeaa01", "NYA", player_war=9.0, salary=40_000_000),
        ],
        player_rows=[
            {"player_id": "judgeaa01", "name_full": "Aaron Judge", "name_first": "Aaron", "name_last": "Judge"},
        ],
        team_rows=[
            {"team_id": "NYA", "team_name": "Yankees"},
            {"team_id": "NYA", "team_name": "Yankees"},
        ],
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["player_id"] == "judgeaa01"
    assert row["team_name"] == "Yankees"
    assert row["player_war"] == 9.0
    assert row["salary"] == 40_000_000
