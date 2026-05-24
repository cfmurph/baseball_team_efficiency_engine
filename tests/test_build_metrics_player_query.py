from __future__ import annotations

import duckdb
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY


@pytest.fixture
def player_metrics_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
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
            team_key VARCHAR,
            team_id VARCHAR,
            franchise_id VARCHAR,
            team_name VARCHAR,
            league_id VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE fact_player_season (
            player_id VARCHAR,
            season_key INTEGER,
            team_id VARCHAR,
            player_type VARCHAR,
            pa DOUBLE,
            hr DOUBLE,
            bb DOUBLE,
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
    try:
        yield con
    finally:
        con.close()


def test_player_query_aggregates_traded_player_without_team_history_fanout(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("traded-player", "Alex Switch", "Alex", "Switch"),
            ("same-name-a", "Chris Lee", "Chris", "Lee"),
            ("same-name-b", "Chris Lee", "Chris", "Lee"),
        ],
    )
    con.executemany(
        "INSERT INTO dim_team VALUES (?, ?, ?, ?, ?)",
        [
            ("AAA_2023", "AAA", "FA", "Old Anchors", "AL"),
            ("AAA_2024", "AAA", "FA", "Alpha Anchors", "AL"),
            ("BBB_2024", "BBB", "FB", "Beta Bears", "NL"),
            ("CCC_2024", "CCC", "FC", "City Cats", "NL"),
        ],
    )
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "traded-player",
                2024,
                "AAA",
                "batter",
                100.0,
                10.0,
                20.0,
                0.400,
                1.0,
                10.0,
                5.00,
                6.00,
                0.0,
                1.0,
                1_000_000.0,
                2_000_000.0,
                "depth",
            ),
            (
                "traded-player",
                2024,
                "BBB",
                "both",
                300.0,
                20.0,
                30.0,
                0.300,
                2.0,
                50.0,
                3.00,
                4.00,
                1.0,
                3.0,
                2_000_000.0,
                6_000_000.0,
                "starter",
            ),
            (
                "same-name-a",
                2024,
                "CCC",
                "batter",
                50.0,
                5.0,
                5.0,
                0.310,
                0.5,
                0.0,
                None,
                None,
                0.0,
                0.5,
                500_000.0,
                1_000_000.0,
                "pre_arbitration",
            ),
            (
                "same-name-b",
                2024,
                "CCC",
                "batter",
                60.0,
                6.0,
                6.0,
                0.320,
                0.6,
                0.0,
                None,
                None,
                0.0,
                0.6,
                600_000.0,
                1_200_000.0,
                "pre_arbitration",
            ),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    traded = result[result["player_id"] == "traded-player"]
    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["team_id"] == "BBB"
    assert row["team_name"] == "Beta Bears"
    assert row["player_type"] == "both"
    assert row["pa"] == pytest.approx(400.0)
    assert row["hr"] == pytest.approx(30.0)
    assert row["bb"] == pytest.approx(50.0)
    assert row["woba"] == pytest.approx(((0.400 * 100.0) + (0.300 * 300.0)) / 400.0)
    assert row["ip"] == pytest.approx(60.0)
    assert row["fip"] == pytest.approx(((5.00 * 10.0) + (3.00 * 50.0)) / 60.0)
    assert row["era"] == pytest.approx(((6.00 * 10.0) + (4.00 * 50.0)) / 60.0)
    assert row["player_war"] == pytest.approx(4.0)
    assert row["salary"] == pytest.approx(3_000_000.0)
    assert row["surplus_value"] == pytest.approx(8_000_000.0)
    assert row["contract_label"] == "starter"


def test_player_query_preserves_same_name_players_by_player_id(
    player_metrics_connection: duckdb.DuckDBPyConnection,
) -> None:
    con = player_metrics_connection
    con.executemany(
        "INSERT INTO dim_player VALUES (?, ?, ?, ?)",
        [
            ("same-name-a", "Chris Lee", "Chris", "Lee"),
            ("same-name-b", "Chris Lee", "Chris", "Lee"),
        ],
    )
    con.execute("INSERT INTO dim_team VALUES ('CCC_2024', 'CCC', 'FC', 'City Cats', 'NL')")
    con.executemany(
        "INSERT INTO fact_player_season VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "same-name-a",
                2024,
                "CCC",
                "batter",
                50.0,
                5.0,
                5.0,
                0.310,
                0.5,
                0.0,
                None,
                None,
                0.0,
                0.5,
                500_000.0,
                1_000_000.0,
                "pre_arbitration",
            ),
            (
                "same-name-b",
                2024,
                "CCC",
                "batter",
                60.0,
                6.0,
                6.0,
                0.320,
                0.6,
                0.0,
                None,
                None,
                0.0,
                0.6,
                600_000.0,
                1_200_000.0,
                "pre_arbitration",
            ),
        ],
    )

    result = con.execute(_PLAYER_QUERY).fetchdf()

    same_name_rows = result[result["name_full"] == "Chris Lee"]
    assert len(same_name_rows) == 2
    assert set(same_name_rows["player_id"]) == {"same-name-a", "same-name-b"}
