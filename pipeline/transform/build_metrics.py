from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


_TEAM_QUERY = """
SELECT
    s.year_id,
    t.team_name,
    t.team_id,
    t.franchise_id,
    t.league_id,
    f.wins,
    f.losses,
    f.games,
    f.runs_scored,
    f.runs_allowed,
    f.run_diff,
    f.pythag_wins,
    f.pythag_gap,
    f.base_runs,
    f.base_runs_gap,
    f.team_batting_war,
    f.team_pitching_war,
    f.team_total_war,
    f.war_win_gap,
    f.payroll,
    f.max_salary,
    f.median_salary,
    f.top_1_salary_share,
    f.top_3_salary_share,
    f.top_5_salary_share,
    f.gini_salary,
    f.dead_money_share,
    f.payroll_per_win,
    f.wins_per_10m,
    f.run_diff_per_10m,
    f.cost_per_war,
    f.war_per_1m,
    f.surplus_value,
    f.window_phase
FROM fact_team_season f
JOIN dim_team t USING (team_key)
JOIN dim_season s USING (season_key)
ORDER BY s.year_id, t.team_name
"""

_PLAYER_QUERY = """
SELECT
    p.player_id,
    dp.name_full,
    dp.name_first,
    dp.name_last,
    p.season_key   AS year_id,
    p.team_id,
    t.team_name,
    p.player_type,
    p.pa,
    p.hr,
    p.woba,
    p.batting_war,
    p.ip,
    p.fip,
    p.era,
    p.pitching_war,
    p.player_war,
    p.salary,
    p.surplus_value,
    p.contract_label
FROM fact_player_season p
LEFT JOIN dim_player dp USING (player_id)
LEFT JOIN dim_team t   ON t.team_id = p.team_id
ORDER BY p.season_key, p.player_war DESC
"""


def _efficiency_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["efficiency_label"] = pd.cut(
        df["wins_per_10m"],
        bins=[-float("inf"), 0.5, 1.0, 1.5, float("inf")],
        labels=["low", "below_avg", "above_avg", "elite"],
    )
    return df


def _top_value_players(player_df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    return (
        player_df[player_df["player_war"] > 0]
        .nlargest(n, "surplus_value")
        .reset_index(drop=True)
    )


def _worst_contracts(player_df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    return (
        player_df[player_df["salary"] > 500_000]
        .nsmallest(n, "surplus_value")
        .reset_index(drop=True)
    )


def _dead_money_leaders(player_df: pd.DataFrame) -> pd.DataFrame:
    return (
        player_df[player_df["contract_label"] == "dead_money"]
        .sort_values("salary", ascending=False)
        .reset_index(drop=True)
    )


def _window_summary(team_df: pd.DataFrame) -> pd.DataFrame:
    """Most recent window phase per franchise."""
    latest = (
        team_df
        .sort_values("year_id")
        .groupby("team_name", as_index=False)
        .last()
        [["team_name", "year_id", "window_phase", "wins", "payroll", "team_total_war"]]
    )
    return latest


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    log.info("Querying team metrics")
    team_df = con.execute(_TEAM_QUERY).fetchdf()

    log.info("Querying player metrics")
    player_df = con.execute(_PLAYER_QUERY).fetchdf()

    con.close()

    # ---- Team exports ----
    team_df.to_csv(artifacts_dir / "team_onfield_contract_metrics.csv", index=False)
    log.info("Wrote team_onfield_contract_metrics.csv (%d rows)", len(team_df))

    efficiency = _efficiency_labels(team_df)
    efficiency.to_csv(artifacts_dir / "team_efficiency_frontier.csv", index=False)

    # Win projection features
    feat_cols = [
        "year_id", "team_name",
        "wins", "run_diff", "pythag_wins", "pythag_gap",
        "base_runs", "base_runs_gap",
        "team_total_war", "war_win_gap",
        "payroll", "max_salary", "median_salary",
        "top_1_salary_share", "top_3_salary_share", "top_5_salary_share",
        "gini_salary", "dead_money_share",
        "payroll_per_win", "wins_per_10m", "run_diff_per_10m",
        "cost_per_war", "war_per_1m", "surplus_value",
    ]
    feat_cols = [c for c in feat_cols if c in team_df.columns]
    team_df[feat_cols].to_csv(artifacts_dir / "win_projection_features.csv", index=False)

    window_df = _window_summary(team_df)
    window_df.to_csv(artifacts_dir / "team_window_phases.csv", index=False)
    log.info("Wrote team_window_phases.csv (%d rows)", len(window_df))

    # ---- Player exports ----
    player_df.to_csv(artifacts_dir / "player_season_metrics.csv", index=False)
    log.info("Wrote player_season_metrics.csv (%d rows)", len(player_df))

    top_value = _top_value_players(player_df)
    top_value.to_csv(artifacts_dir / "player_top_surplus_value.csv", index=False)

    worst = _worst_contracts(player_df)
    worst.to_csv(artifacts_dir / "player_worst_contracts.csv", index=False)

    dead = _dead_money_leaders(player_df)
    dead.to_csv(artifacts_dir / "player_dead_money.csv", index=False)
    log.info("Wrote contract analysis CSVs")

    typer.echo(f"Wrote all artifacts to {artifacts_dir}")


if __name__ == "__main__":
    app()
