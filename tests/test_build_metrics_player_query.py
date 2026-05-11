from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(":memory:")
    connection.execute(WAREHOUSE_DDL)
    try:
        yield connection
    finally:
        connection.close()


def _insert_player(
    con: duckdb.DuckDBPyConnection,
    player_id: str,
    name_full: str,
    *,
    name_first: str = "Test",
    name_last: str = "Player",
) -> None:
    con.execute(
        """
        INSERT INTO dim_player (
            player_id, name_first, name_last, name_full,
            birth_year, birth_country, throws, bats
        )
        VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL)
        """,
        [player_id, name_first, name_last, name_full],
    )


def _insert_team(
    con: duckdb.DuckDBPyConnection,
    team_key: str,
    team_id: str,
    team_name: str,
    *,
    franchise_id: str = "FR",
    league_id: str = "AL",
) -> None:
    con.execute(
        """
        INSERT INTO dim_team (team_key, team_id, franchise_id, team_name, league_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        [team_key, team_id, franchise_id, team_name, league_id],
    )


def _insert_fact_player_season(
    con: duckdb.DuckDBPyConnection,
    player_id: str,
    season_key: int,
    team_id: str,
    player_type: str,
    *,
    pa: float | None = None,
    hr: float | None = None,
    bb: float | None = None,
    woba: float | None = None,
    batting_war: float | None = None,
    ip: float | None = None,
    fip: float | None = None,
    era: float | None = None,
    pitching_war: float | None = None,
    player_war: float | None = None,
    salary: float | None = None,
    surplus_value: float | None = None,
    contract_label: str | None = None,
) -> None:
    con.execute(
        """
        INSERT INTO fact_player_season (
            player_id, season_key, team_id, player_type,
            pa, hr, bb, woba, batting_war,
            ip, fip, era, pitching_war,
            player_war, salary, surplus_value, contract_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            player_id,
            season_key,
            team_id,
            player_type,
            pa,
            hr,
            bb,
            woba,
            batting_war,
            ip,
            fip,
            era,
            pitching_war,
            player_war,
            salary,
            surplus_value,
            contract_label,
        ],
    )


def test_player_query_aggregates_traded_player_at_player_season_grain(
    con: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player(con, "player-traded", "Traded Star")
    _insert_team(con, "AAA_2024", "AAA", "Alpha")
    _insert_team(con, "BBB_2024", "BBB", "Bravo")

    _insert_fact_player_season(
        con,
        "player-traded",
        2024,
        "AAA",
        "batter",
        pa=100,
        hr=10,
        bb=20,
        woba=0.300,
        batting_war=1.0,
        ip=10,
        fip=4.00,
        era=5.00,
        pitching_war=0.2,
        player_war=1.2,
        salary=1_000_000,
        surplus_value=500_000,
        contract_label="fair_value",
    )
    _insert_fact_player_season(
        con,
        "player-traded",
        2024,
        "BBB",
        "pitcher",
        pa=300,
        hr=20,
        bb=40,
        woba=0.400,
        batting_war=2.0,
        ip=90,
        fip=2.00,
        era=3.00,
        pitching_war=1.8,
        player_war=3.8,
        salary=2_000_000,
        surplus_value=1_500_000,
        contract_label="surplus_value",
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "player-traded"
    assert row["year_id"] == 2024
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Bravo"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == pytest.approx(400)
    assert row["hr"] == pytest.approx(30)
    assert row["bb"] == pytest.approx(60)
    assert row["woba"] == pytest.approx(((0.300 * 100) + (0.400 * 300)) / 400)
    assert row["batting_war"] == pytest.approx(3.0)
    assert row["ip"] == pytest.approx(100)
    assert row["fip"] == pytest.approx(((4.00 * 10) + (2.00 * 90)) / 100)
    assert row["era"] == pytest.approx(((5.00 * 10) + (3.00 * 90)) / 100)
    assert row["pitching_war"] == pytest.approx(2.0)
    assert row["player_war"] == pytest.approx(5.0)
    assert row["salary"] == pytest.approx(3_000_000)
    assert row["surplus_value"] == pytest.approx(2_000_000)
    assert row["contract_label"] == "surplus_value"


def test_player_query_joins_dim_team_by_season_specific_team_key(
    con: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player(con, "player-one", "Stable Player")
    _insert_team(con, "AAA_2023", "AAA", "Old Alpha")
    _insert_team(con, "AAA_2024", "AAA", "New Alpha")

    _insert_fact_player_season(
        con,
        "player-one",
        2024,
        "AAA",
        "batter",
        pa=120,
        hr=8,
        bb=15,
        woba=0.350,
        batting_war=1.5,
        player_war=1.5,
        salary=900_000,
        surplus_value=250_000,
        contract_label="fair_value",
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "New Alpha"
    assert row["pa"] == pytest.approx(120)
    assert row["player_war"] == pytest.approx(1.5)


def test_player_query_keeps_same_name_players_distinct_by_player_id(
    con: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player(con, "alex-a", "Alex Gonzalez")
    _insert_player(con, "alex-b", "Alex Gonzalez")
    _insert_team(con, "AAA_2024", "AAA", "Alpha")

    _insert_fact_player_season(
        con,
        "alex-a",
        2024,
        "AAA",
        "batter",
        pa=300,
        hr=12,
        bb=30,
        woba=0.330,
        batting_war=2.0,
        player_war=2.0,
        salary=1_000_000,
        surplus_value=400_000,
        contract_label="surplus_value",
    )
    _insert_fact_player_season(
        con,
        "alex-b",
        2024,
        "AAA",
        "batter",
        pa=250,
        hr=5,
        bb=20,
        woba=0.290,
        batting_war=0.4,
        player_war=0.4,
        salary=750_000,
        surplus_value=-100_000,
        contract_label="fair_value",
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    assert sorted(result["player_id"].tolist()) == ["alex-a", "alex-b"]
    assert (result["name_full"] == "Alex Gonzalez").all()
