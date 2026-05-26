from __future__ import annotations

import duckdb
import pandas as pd

from pipeline.transform import build_metrics


def test_efficiency_labels_bucket_wins_per_10m() -> None:
    df = pd.DataFrame({"wins_per_10m": [0.3, 0.8, 1.2, 2.0]})

    result = build_metrics._efficiency_labels(df)

    assert result["efficiency_label"].astype(str).tolist() == [
        "low",
        "below_avg",
        "above_avg",
        "elite",
    ]


def test_contract_exports_filter_and_sort_material_contract_risks() -> None:
    player_df = pd.DataFrame(
        {
            "name_full": ["Value Bat", "Replacement", "Injured Ace", "Bench Deal"],
            "player_war": [4.0, 0.0, -0.5, 1.0],
            "salary": [2_000_000, 1_000_000, 25_000_000, 500_000],
            "surplus_value": [28_000_000, 5_000_000, -24_000_000, -2_000_000],
            "contract_label": ["surplus_value", "fair_value", "dead_money", "overpaid"],
        }
    )

    top_value = build_metrics._top_value_players(player_df, n=3)
    worst = build_metrics._worst_contracts(player_df, n=3)
    dead_money = build_metrics._dead_money_leaders(player_df)

    assert top_value["name_full"].tolist() == ["Value Bat", "Bench Deal"]
    assert worst["name_full"].tolist() == ["Injured Ace", "Replacement", "Value Bat"]
    assert dead_money["name_full"].tolist() == ["Injured Ace"]


def test_window_summary_keeps_latest_phase_per_team() -> None:
    team_df = pd.DataFrame(
        {
            "team_name": ["Aces", "Bears", "Aces", "Bears"],
            "year_id": [2022, 2021, 2024, 2023],
            "window_phase": ["building", "retooling", "contending", "rebuilding"],
            "wins": [78, 85, 96, 67],
            "payroll": [90_000_000, 120_000_000, 150_000_000, 80_000_000],
            "team_total_war": [30.0, 35.0, 45.0, 22.0],
        }
    )

    result = build_metrics._window_summary(team_df).sort_values("team_name").reset_index(drop=True)

    assert result[["team_name", "year_id", "window_phase", "wins"]].to_dict("records") == [
        {"team_name": "Aces", "year_id": 2024, "window_phase": "contending", "wins": 96},
        {"team_name": "Bears", "year_id": 2023, "window_phase": "rebuilding", "wins": 67},
    ]


def test_table_has_rows_handles_present_empty_and_missing_tables() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE present_empty (id INTEGER)")
        con.execute("CREATE TABLE present_with_rows (id INTEGER)")
        con.execute("INSERT INTO present_with_rows VALUES (1)")

        assert build_metrics._table_has_rows(con, "present_empty") is False
        assert build_metrics._table_has_rows(con, "present_with_rows") is True
        assert build_metrics._table_has_rows(con, "missing_table") is False
    finally:
        con.close()


def test_player_query_aggregates_traded_player_to_one_season_row() -> None:
    con = _player_query_connection()
    try:
        result = con.execute(build_metrics._PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    traded = result[result["player_id"] == "traded-1"].iloc[0]

    assert result["player_id"].tolist().count("traded-1") == 1
    assert traded["year_id"] == 2024
    assert traded["team_id"] == "BOS"
    assert traded["team_name"] == "Boston Red Sox"
    assert traded["player_type"] == "both"
    assert traded["pa"] == 150
    assert traded["ip"] == 30.0
    assert traded["player_war"] == 4.0
    assert traded["salary"] == 4_000_000
    assert traded["surplus_value"] == 18_000_000
    assert traded["contract_label"] == "surplus_value"


def test_player_query_deduplicates_repeated_team_dimension_rows() -> None:
    con = _player_query_connection()
    try:
        result = con.execute(build_metrics._PLAYER_QUERY).fetchdf()
    finally:
        con.close()

    steady = result[result["player_id"] == "steady-1"].iloc[0]

    assert len(result) == 2
    assert steady["team_id"] == "NYY"
    assert steady["team_name"] == "New York Yankees"
    assert steady["player_war"] == 2.5
    assert steady["salary"] == 8_000_000


def _player_query_connection() -> duckdb.DuckDBPyConnection:
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
            team_id VARCHAR,
            team_name VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE fact_player_season (
            player_id VARCHAR,
            team_id VARCHAR,
            season_key INTEGER,
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
            salary INTEGER,
            surplus_value INTEGER,
            contract_label VARCHAR
        )
        """
    )
    con.execute(
        """
        INSERT INTO dim_player VALUES
            ('traded-1', 'Traded Star', 'Traded', 'Star'),
            ('steady-1', 'Steady Pitcher', 'Steady', 'Pitcher')
        """
    )
    con.execute(
        """
        INSERT INTO dim_team VALUES
            ('NYY', 'New York Yankees'),
            ('NYY', 'New York Yankees'),
            ('NYY', 'New York Yankees'),
            ('BOS', 'Boston Red Sox'),
            ('OAK', 'Oakland Athletics')
        """
    )
    con.execute(
        """
        INSERT INTO fact_player_season VALUES
            ('traded-1', 'OAK', 2024, 'batter', 100, 5, 10, 0.310, 1.0, 0.0, NULL, NULL, 0.0, 1.0, 1_000_000, 3_000_000, 'fair_value'),
            ('traded-1', 'BOS', 2024, 'both', 50, 7, 8, 0.410, 2.0, 30.0, 3.20, 2.80, 1.0, 3.0, 3_000_000, 15_000_000, 'surplus_value'),
            ('steady-1', 'NYY', 2024, 'pitcher', 0, 0, 0, NULL, 0.0, 160.0, 3.70, 3.30, 2.5, 2.5, 8_000_000, 6_000_000, 'fair_value')
        """
    )
    return con
