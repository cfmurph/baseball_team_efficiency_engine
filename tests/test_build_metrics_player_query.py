from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture
def warehouse() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("trade01", "Traded", "Star", "Traded Star", 1990, "USA", "R", "R"),
            ("youngch03", "Chris", "Young", "Chris Young", 1979, "USA", "R", "R"),
            ("youngch04", "Chris", "Young", "Chris Young", 1983, "USA", "R", "R"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("NYA_2020", "NYA", "NYY", "New York Yankees", "AL"),
            ("BOS_2020", "BOS", "BOS", "Boston Red Sox", "AL"),
            ("BOS_2021", "BOS", "BOS", "Boston Red Sox", "AL"),
            ("ARI_2020", "ARI", "ARI", "Arizona Diamondbacks", "NL"),
            ("NYN_2020", "NYN", "NYM", "New York Mets", "NL"),
        ],
    )
    yield con
    con.close()


def _insert_player_fact(
    con: duckdb.DuckDBPyConnection,
    *,
    player_id: str,
    season_key: int,
    team_id: str,
    player_type: str = "batter",
    pa: float = 0.0,
    hr: float = 0.0,
    bb: float = 0.0,
    woba: float | None = None,
    batting_war: float = 0.0,
    ip: float = 0.0,
    fip: float | None = None,
    era: float | None = None,
    pitching_war: float = 0.0,
    player_war: float = 0.0,
    salary: float = 0.0,
    surplus_value: float = 0.0,
    contract_label: str = "fair_value",
) -> None:
    con.execute(
        """
        INSERT INTO fact_player_season VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
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


def test_player_query_collapses_traded_player_to_one_season_row(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player_fact(
        warehouse,
        player_id="trade01",
        season_key=2020,
        team_id="NYA",
        player_type="batter",
        pa=100,
        hr=10,
        bb=20,
        woba=0.340,
        batting_war=1.5,
        player_war=1.5,
        salary=2_000_000,
        surplus_value=10_000_000,
        contract_label="fair_value",
    )
    _insert_player_fact(
        warehouse,
        player_id="trade01",
        season_key=2020,
        team_id="BOS",
        player_type="pitcher",
        pa=50,
        hr=5,
        bb=10,
        woba=0.380,
        batting_war=0.5,
        ip=80,
        fip=3.50,
        era=3.20,
        pitching_war=3.0,
        player_war=3.5,
        salary=3_000_000,
        surplus_value=20_000_000,
        contract_label="surplus_value",
    )

    result = warehouse.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["player_id"] == "trade01"
    assert row["name_full"] == "Traded Star"
    assert row["year_id"] == 2020
    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["contract_label"] == "surplus_value"
    assert row["player_type"] == "pitcher"
    assert row["pa"] == pytest.approx(150)
    assert row["hr"] == pytest.approx(15)
    assert row["bb"] == pytest.approx(30)
    assert row["batting_war"] == pytest.approx(2.0)
    assert row["ip"] == pytest.approx(80)
    assert row["pitching_war"] == pytest.approx(3.0)
    assert row["player_war"] == pytest.approx(5.0)
    assert row["salary"] == pytest.approx(5_000_000)
    assert row["surplus_value"] == pytest.approx(30_000_000)


def test_player_query_does_not_fan_out_when_team_history_repeats_team_id(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player_fact(
        warehouse,
        player_id="trade01",
        season_key=2020,
        team_id="BOS",
        player_war=2.0,
        salary=1_000_000,
        surplus_value=15_000_000,
    )

    result = warehouse.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 1
    row = result.iloc[0]
    assert row["team_name"] == "Boston Red Sox"
    assert row["player_war"] == pytest.approx(2.0)
    assert row["salary"] == pytest.approx(1_000_000)


def test_player_query_keeps_same_name_players_distinct_by_player_id(
    warehouse: duckdb.DuckDBPyConnection,
) -> None:
    _insert_player_fact(
        warehouse,
        player_id="youngch03",
        season_key=2020,
        team_id="ARI",
        player_war=2.1,
        salary=700_000,
        surplus_value=16_000_000,
    )
    _insert_player_fact(
        warehouse,
        player_id="youngch04",
        season_key=2020,
        team_id="NYN",
        player_war=1.2,
        salary=600_000,
        surplus_value=9_000_000,
    )

    result = warehouse.execute(_PLAYER_QUERY).fetchdf()

    same_name = result[result["name_full"] == "Chris Young"].sort_values("player_id")
    assert same_name["player_id"].tolist() == ["youngch03", "youngch04"]
    assert same_name["team_name"].tolist() == ["Arizona Diamondbacks", "New York Mets"]
    assert same_name["player_war"].tolist() == pytest.approx([2.1, 1.2])
