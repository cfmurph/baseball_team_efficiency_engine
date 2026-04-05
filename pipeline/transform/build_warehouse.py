from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import read_csv, ensure_dir
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.metrics import pythagorean_wins, safe_divide, top_salary_shares, salary_concentration

app = typer.Typer(add_completion=False)


def build_team_metrics(settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_year = settings["min_year"]
    raw_dir = Path(settings["raw_dir"])

    teams = read_csv(raw_dir / "teams.csv")
    salaries = read_csv(raw_dir / "salaries.csv")

    teams = teams[(teams["yearID"] >= min_year) & (teams["lgID"].isin(["AL", "NL"]))].copy()
    teams["team_key"] = teams["teamID"].astype(str) + "_" + teams["yearID"].astype(str)
    teams["season_key"] = teams["yearID"]
    teams["run_diff"] = teams["R"] - teams["RA"]
    teams["pythag_wins"] = pythagorean_wins(teams["R"], teams["RA"], teams["G"])

    salary_grouped = (
        salaries[salaries["yearID"] >= min_year]
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(payroll=("salary", "sum"), max_salary=("salary", "max"), median_salary=("salary", "median"))
    )

    salary_shares = (
        salaries[salaries["yearID"] >= min_year]
        .groupby(["yearID", "teamID"], as_index=False)
        .apply(lambda g: pd.concat([top_salary_shares(g), salary_concentration(g)]), include_groups=False)
        .reset_index()
        .rename(columns={"yearID": "yearID", "teamID": "teamID"})
    )
    if "level_2" in salary_shares.columns:
        salary_shares = salary_shares.drop(columns=["level_2"])

    team_metrics = teams.merge(salary_grouped, on=["yearID", "teamID"], how="left")
    team_metrics = team_metrics.merge(salary_shares, on=["yearID", "teamID"], how="left")

    team_metrics["payroll_per_win"] = safe_divide(team_metrics["payroll"], team_metrics["W"])
    team_metrics["wins_per_10m"] = safe_divide(team_metrics["W"] * 10_000_000, team_metrics["payroll"])
    team_metrics["run_diff_per_10m"] = safe_divide(team_metrics["run_diff"] * 10_000_000, team_metrics["payroll"])

    dim_team = (
        teams[["team_key", "teamID", "franchID", "name", "lgID"]]
        .drop_duplicates()
        .rename(columns={"teamID": "team_id", "franchID": "franchise_id", "name": "team_name", "lgID": "league_id"})
    )
    dim_season = teams[["season_key", "yearID"]].drop_duplicates().rename(columns={"yearID": "year_id"})
    fact_salary = salaries[salaries["yearID"] >= min_year][["yearID", "teamID", "playerID", "salary"]].rename(
        columns={"yearID": "season_key", "teamID": "team_id", "playerID": "player_id"}
    )

    fact_team_season = team_metrics.rename(
        columns={
            "teamID": "team_id",
            "W": "wins",
            "L": "losses",
            "G": "games",
            "R": "runs_scored",
            "RA": "runs_allowed",
            "SOA": "strikeouts",
            "attend": "attendance",
        }
    )[[
        "team_key",
        "season_key",
        "wins",
        "losses",
        "games",
        "runs_scored",
        "runs_allowed",
        "strikeouts",
        "attendance",
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
    ]]

    return dim_team, dim_season, fact_salary, fact_team_season


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    warehouse_path = Path(settings["warehouse_path"])
    ensure_dir(warehouse_path.parent)

    dim_team, dim_season, fact_salary, fact_team_season = build_team_metrics(settings)

    con = duckdb.connect(str(warehouse_path))
    con.execute(WAREHOUSE_DDL)
    con.register("dim_team_df", dim_team)
    con.register("dim_season_df", dim_season)
    con.register("fact_salary_df", fact_salary)
    con.register("fact_team_season_df", fact_team_season)

    con.execute("DELETE FROM dim_team")
    con.execute("DELETE FROM dim_season")
    con.execute("DELETE FROM fact_salary")
    con.execute("DELETE FROM fact_team_season")

    con.execute("INSERT INTO dim_team SELECT * FROM dim_team_df")
    con.execute("INSERT INTO dim_season SELECT * FROM dim_season_df")
    con.execute("INSERT INTO fact_salary SELECT * FROM fact_salary_df")
    con.execute("INSERT INTO fact_team_season SELECT * FROM fact_team_season_df")
    con.close()

    typer.echo(f"Built warehouse: {warehouse_path}")


if __name__ == "__main__":
    app()
