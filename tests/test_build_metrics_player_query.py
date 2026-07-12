from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from pipeline.transform.build_metrics import _PLAYER_QUERY
from src.baseball_analytics.schema import WAREHOUSE_DDL


@pytest.fixture()
def player_query_df() -> pd.DataFrame:
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.execute("INSERT INTO dim_season VALUES (2024, 2024)")
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYA_2023', 'NYA', 'NYY', 'New York Yankees', 'AL'),
            ('NYA_2024', 'NYA', 'NYY', 'New York Yankees', 'AL'),
            ('BOS_2024', 'BOS', 'BOS', 'Boston Red Sox', 'AL'),
            ('LAN_2024', 'LAN', 'LAD', 'Los Angeles Dodgers', 'NL')
        """
    )
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('traded01', 'Trade', 'Target', 'Trade Target', 1990, 'USA', 'R', 'L'),
            ('single01', 'Single', 'Team', 'Single Team', 1991, 'USA', 'R', 'R'),
            ('smithjo01', 'John', 'Smith', 'John Smith', 1992, 'USA', 'R', 'R'),
            ('smithjo02', 'John', 'Smith', 'John Smith', 1993, 'USA', 'L', 'L'),
            ('twoway01', 'Two', 'Way', 'Two Way', 1994, 'USA', 'R', 'L')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('traded01', 2024, 'NYA', 'batter', 250, 10, 20, 0.330, 2.0, NULL, NULL, NULL, 0.0, 2.0, 1000000, 15000000, 'surplus_value'),
            ('traded01', 2024, 'BOS', 'pitcher', NULL, NULL, NULL, NULL, 0.0, 100, 3.50, 3.20, 5.0, 5.0, 10000000, -5000000, 'overpaid'),
            ('single01', 2024, 'NYA', 'batter', 500, 30, 60, 0.380, 3.0, NULL, NULL, NULL, 0.0, 3.0, 700000, 23300000, 'surplus_value'),
            ('smithjo01', 2024, 'LAN', 'batter', 100, 4, 10, 0.300, 1.0, NULL, NULL, NULL, 0.0, 1.0, 750000, 7250000, 'surplus_value'),
            ('smithjo02', 2024, 'LAN', 'batter', 120, 3, 8, 0.290, 0.5, NULL, NULL, NULL, 0.0, 0.5, 750000, 3250000, 'fair_value'),
            ('twoway01', 2024, 'LAN', 'both', 200, 8, 15, 0.310, 1.5, 50, 3.25, 3.00, 2.0, 3.5, 2000000, 26000000, 'surplus_value')
        """
    )
    try:
        return con.execute(_PLAYER_QUERY).fetchdf()
    finally:
        con.close()


def test_player_query_collapses_traded_player_to_one_season_row(player_query_df: pd.DataFrame) -> None:
    traded = player_query_df[player_query_df["player_id"] == "traded01"]

    assert len(traded) == 1
    row = traded.iloc[0]
    assert row["player_war"] == pytest.approx(7.0)
    assert row["salary"] == pytest.approx(11_000_000)


def test_player_query_uses_highest_war_stint_for_primary_team_and_contract(player_query_df: pd.DataFrame) -> None:
    row = player_query_df[player_query_df["player_id"] == "traded01"].iloc[0]

    assert row["team_id"] == "BOS"
    assert row["team_name"] == "Boston Red Sox"
    assert row["contract_label"] == "overpaid"


def test_player_query_does_not_fan_out_repeated_team_history(player_query_df: pd.DataFrame) -> None:
    row = player_query_df[player_query_df["player_id"] == "single01"].iloc[0]

    assert row["team_name"] == "New York Yankees"
    assert row["player_war"] == pytest.approx(3.0)
    assert row["salary"] == pytest.approx(700_000)


def test_player_query_preserves_same_name_players_by_player_id(player_query_df: pd.DataFrame) -> None:
    smiths = player_query_df[player_query_df["name_full"] == "John Smith"]

    assert set(smiths["player_id"]) == {"smithjo01", "smithjo02"}
    assert len(smiths) == 2


def test_player_query_chooses_most_specific_player_type(player_query_df: pd.DataFrame) -> None:
    two_way = player_query_df[player_query_df["player_id"] == "twoway01"].iloc[0]
    traded = player_query_df[player_query_df["player_id"] == "traded01"].iloc[0]

    assert two_way["player_type"] == "both"
    assert traded["player_type"] == "pitcher"
