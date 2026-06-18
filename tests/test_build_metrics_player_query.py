from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


def _create_player_metrics_fixture(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE fact_player_season (
            player_id VARCHAR,
            season_key INTEGER,
            team_id VARCHAR,
            player_type VARCHAR,
            pa INTEGER,
            hr INTEGER,
            bb INTEGER,
            woba DOUBLE,
            batting_war DOUBLE,
            ip DOUBLE,
            fip DOUBLE,
            era DOUBLE,
            pitching_war DOUBLE,
            player_war DOUBLE,
            salary DOUBLE,
            surplus_value DOUBLE,
            contract_label VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dim_player (
            player_id VARCHAR,
            name_full VARCHAR,
            name_first VARCHAR,
            name_last VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE dim_team (
            team_id VARCHAR,
            team_name VARCHAR
        )
        """
    )

    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?)",
        [
            ("NYA", "New York Yankees"),
            ("NYA", "New York Yankees"),
            ("BOS", "Boston Red Sox"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("switch01", "Pat Switch", "Pat", "Switch"),
            ("smitha01", "Alex Smith", "Alex", "Smith"),
            ("smithb01", "Alex Smith", "Alex", "Smith"),
        ],
    )
    con.executemany(
        """
        INSERT INTO fact_player_season VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        [
            (
                "switch01",
                2010,
                "NYA",
                "batter",
                100,
                10,
                20,
                0.350,
                1.5,
                0.0,
                None,
                None,
                0.0,
                1.5,
                1_000_000.0,
                11_000_000.0,
                "surplus_value",
            ),
            (
                "switch01",
                2010,
                "BOS",
                "batter",
                50,
                5,
                10,
                0.300,
                0.2,
                0.0,
                None,
                None,
                0.0,
                0.2,
                500_000.0,
                1_100_000.0,
                "fair_value",
            ),
            (
                "smitha01",
                2010,
                "NYA",
                "pitcher",
                0,
                0,
                0,
                None,
                0.0,
                100.0,
                3.2,
                3.5,
                2.0,
                2.0,
                2_000_000.0,
                14_000_000.0,
                "surplus_value",
            ),
            (
                "smithb01",
                2010,
                "NYA",
                "pitcher",
                0,
                0,
                0,
                None,
                0.0,
                50.0,
                4.2,
                4.5,
                0.5,
                0.5,
                750_000.0,
                3_250_000.0,
                "fair_value",
            ),
        ],
    )


@pytest.fixture
def player_metrics_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    _create_player_metrics_fixture(con)
    try:
        yield con
    finally:
        con.close()


def test_player_query_collapses_traded_stints_to_one_player_season(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    traded = result[result["player_id"] == "switch01"].iloc[0]
    assert len(result[result["player_id"] == "switch01"]) == 1
    assert traded["year_id"] == 2010
    assert traded["team_id"] == "NYA"
    assert traded["team_name"] == "New York Yankees"
    assert traded["pa"] == 150
    assert traded["hr"] == 15
    assert traded["bb"] == 30
    assert traded["batting_war"] == pytest.approx(1.7)
    assert traded["player_war"] == pytest.approx(1.7)
    assert traded["salary"] == pytest.approx(1_500_000.0)
    assert traded["surplus_value"] == pytest.approx(12_100_000.0)
    assert traded["contract_label"] == "surplus_value"


def test_player_query_dedupes_repeated_team_dimension_rows(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()

    assert len(result) == 3
    assert result["player_id"].value_counts().to_dict() == {
        "switch01": 1,
        "smitha01": 1,
        "smithb01": 1,
    }


def test_player_query_keeps_same_name_players_distinct_by_player_id(
    player_metrics_con: duckdb.DuckDBPyConnection,
) -> None:
    result = player_metrics_con.execute(_PLAYER_QUERY).fetchdf()
    alex_smiths = result[result["name_full"] == "Alex Smith"].sort_values("player_id")

    assert alex_smiths["player_id"].tolist() == ["smitha01", "smithb01"]
    assert alex_smiths["pitching_war"].tolist() == [2.0, 0.5]
