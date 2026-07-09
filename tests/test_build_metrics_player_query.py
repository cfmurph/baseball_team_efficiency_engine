from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


def _player_rows(con: duckdb.DuckDBPyConnection) -> list[dict]:
    return con.execute(_PLAYER_QUERY).fetchdf().to_dict("records")


def test_player_query_collapses_traded_stints_without_team_history_fanout() -> None:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("NYY_2023", "NYA", "NYY", "New York Yankees", "AL"),
            ("NYY_2024", "NYA", "NYY", "New York Yankees", "AL"),
            ("BOS_2024", "BOS", "BOS", "Boston Red Sox", "AL"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("traded01", "Taylor", "Traded", "Taylor Traded", 1995, "USA", "R", "L"),
            ("single01", "Sam", "Single", "Sam Single", 1994, "USA", "R", "R"),
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
                "traded01",
                2024,
                "NYA",
                "batter",
                300.0,
                10.0,
                40.0,
                0.350,
                2.0,
                0.0,
                None,
                None,
                0.0,
                2.0,
                8_000_000.0,
                8_000_000.0,
                "fair_value",
            ),
            (
                "traded01",
                2024,
                "BOS",
                "pitcher",
                150.0,
                4.0,
                20.0,
                0.320,
                0.8,
                50.0,
                3.10,
                3.50,
                3.0,
                3.8,
                2_000_000.0,
                28_000_000.0,
                "surplus_value",
            ),
            (
                "single01",
                2024,
                "NYA",
                "batter",
                500.0,
                20.0,
                70.0,
                0.370,
                4.5,
                0.0,
                None,
                None,
                0.0,
                4.5,
                12_000_000.0,
                24_000_000.0,
                "fair_value",
            ),
        ],
    )

    rows = _player_rows(con)

    assert {(row["player_id"], row["year_id"]) for row in rows} == {
        ("traded01", 2024),
        ("single01", 2024),
    }
    traded = next(row for row in rows if row["player_id"] == "traded01")
    assert traded["pa"] == pytest.approx(450.0)
    assert traded["hr"] == pytest.approx(14.0)
    assert traded["player_war"] == pytest.approx(5.8)
    assert traded["salary"] == pytest.approx(10_000_000.0)
    assert traded["team_id"] == "BOS"
    assert traded["team_name"] == "Boston Red Sox"
    assert traded["contract_label"] == "surplus_value"
    assert traded["player_type"] == "pitcher"


def test_player_query_preserves_same_name_players_as_distinct_people() -> None:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.execute(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        ["BOS_2024", "BOS", "BOS", "Boston Red Sox", "AL"],
    )
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("smith01", "Alex", "Smith", "Alex Smith", 1992, "USA", "R", "R"),
            ("smith02", "Alex", "Smith", "Alex Smith", 1998, "USA", "L", "L"),
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
                "smith01",
                2024,
                "BOS",
                "batter",
                250.0,
                8.0,
                30.0,
                0.310,
                1.2,
                0.0,
                None,
                None,
                0.0,
                1.2,
                1_000_000.0,
                8_600_000.0,
                "surplus_value",
            ),
            (
                "smith02",
                2024,
                "BOS",
                "pitcher",
                0.0,
                0.0,
                0.0,
                None,
                0.0,
                80.0,
                3.20,
                3.80,
                2.4,
                2.4,
                3_000_000.0,
                16_200_000.0,
                "surplus_value",
            ),
        ],
    )

    rows = _player_rows(con)

    assert [row["player_id"] for row in rows] == ["smith02", "smith01"]
    assert all(row["name_full"] == "Alex Smith" for row in rows)
    assert {row["player_id"]: row["player_type"] for row in rows} == {
        "smith01": "batter",
        "smith02": "pitcher",
    }
