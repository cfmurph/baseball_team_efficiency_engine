from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    team_metrics = con.execute(
        """
        SELECT f.*, t.team_name, t.team_id, t.franchise_id, t.league_id, s.year_id
        FROM fact_team_season f
        JOIN dim_team t USING(team_key)
        JOIN dim_season s USING(season_key)
        ORDER BY year_id, team_name
        """
    ).fetchdf()

    frontier = team_metrics[["year_id", "team_name", "payroll", "wins", "run_diff", "wins_per_10m", "payroll_per_win"]].copy()
    frontier["efficiency_label"] = pd.cut(
        frontier["wins_per_10m"],
        bins=[-float("inf"), 0.5, 1.0, 1.5, float("inf")],
        labels=["low", "below_avg", "above_avg", "elite"],
    )

    features = team_metrics[[
        "year_id",
        "team_name",
        "wins",
        "run_diff",
        "pythag_wins",
        "payroll",
        "max_salary",
        "median_salary",
        "top_1_salary_share",
        "top_3_salary_share",
        "top_5_salary_share",
        "gini_salary",
        "payroll_per_win",
        "wins_per_10m",
        "run_diff_per_10m",
    ]].copy()
    features["pythag_gap"] = features["wins"] - features["pythag_wins"]

    team_metrics.to_csv(Path(artifacts_dir) / "team_onfield_contract_metrics.csv", index=False)
    frontier.to_csv(Path(artifacts_dir) / "team_efficiency_frontier.csv", index=False)
    features.to_csv(Path(artifacts_dir) / "win_projection_features.csv", index=False)
    con.close()

    typer.echo(f"Wrote artifacts to {artifacts_dir}")


if __name__ == "__main__":
    app()
