from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import read_csv, ensure_dir
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.metrics import (
    pythagorean_wins,
    safe_divide,
    top_salary_shares,
    salary_concentration,
    cost_per_war,
    war_per_dollar,
    surplus_value_team,
    pythag_gap,
    war_win_gap,
    player_surplus_value,
    classify_contract,
    payroll_underperformer_share,
    detect_team_window,
)
from src.baseball_analytics.war import batting_war, pitching_war, team_war_totals, base_runs

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)


# ---------------------------------------------------------------------------
# Dimension builders
# ---------------------------------------------------------------------------

def build_dim_team(teams: pd.DataFrame) -> pd.DataFrame:
    return (
        teams[["team_key", "teamID", "franchID", "name", "lgID"]]
        .drop_duplicates()
        .rename(
            columns={
                "teamID": "team_id",
                "franchID": "franchise_id",
                "name": "team_name",
                "lgID": "league_id",
            }
        )
    )


def build_dim_season(teams: pd.DataFrame) -> pd.DataFrame:
    return (
        teams[["season_key", "yearID"]]
        .drop_duplicates()
        .rename(columns={"yearID": "year_id"})
    )


def build_dim_player(people: pd.DataFrame) -> pd.DataFrame:
    p = people.copy()
    # Lahman People columns vary by version; guard missing ones
    for col in ["nameFirst", "nameLast", "birthYear", "birthCountry", "throws", "bats"]:
        if col not in p.columns:
            p[col] = None
    p["name_full"] = p["nameFirst"].fillna("") + " " + p["nameLast"].fillna("")
    return (
        p[["playerID", "nameFirst", "nameLast", "name_full", "birthYear", "birthCountry", "throws", "bats"]]
        .drop_duplicates("playerID")
        .rename(
            columns={
                "playerID": "player_id",
                "nameFirst": "name_first",
                "nameLast": "name_last",
                "birthYear": "birth_year",
                "birthCountry": "birth_country",
            }
        )
    )


# ---------------------------------------------------------------------------
# Fact: salary
# ---------------------------------------------------------------------------

def build_fact_salary(salaries: pd.DataFrame, min_year: int) -> pd.DataFrame:
    return (
        salaries[salaries["yearID"] >= min_year][["yearID", "teamID", "playerID", "salary"]]
        .rename(
            columns={
                "yearID": "season_key",
                "teamID": "team_id",
                "playerID": "player_id",
            }
        )
        .drop_duplicates(["season_key", "team_id", "player_id"])
    )


# ---------------------------------------------------------------------------
# Fact: player season
# ---------------------------------------------------------------------------

def build_fact_player_season(
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    salaries: pd.DataFrame,
    min_year: int,
) -> pd.DataFrame:
    import numpy as np  # noqa: F401
    bat_war = batting_war(batting[batting["yearID"] >= min_year])
    pit_war = pitching_war(pitching[pitching["yearID"] >= min_year])

    # Merge bat + pit
    bat_war = bat_war.rename(columns={"yearID": "season_key", "teamID": "team_id", "playerID": "player_id"})
    pit_war = pit_war.rename(columns={"yearID": "season_key", "teamID": "team_id", "playerID": "player_id"})

    player_df = pd.merge(
        bat_war,
        pit_war,
        on=["player_id", "season_key", "team_id"],
        how="outer",
        suffixes=("_bat", "_pit"),
    )

    # Derive player_type
    has_bat = player_df["batting_war"].notna()
    has_pit = player_df["pitching_war"].notna()
    player_df["player_type"] = "batter"
    player_df.loc[has_pit & ~has_bat, "player_type"] = "pitcher"
    player_df.loc[has_pit & has_bat, "player_type"] = "both"

    player_df["batting_war"] = player_df["batting_war"].fillna(0)
    player_df["pitching_war"] = player_df["pitching_war"].fillna(0)
    player_df["player_war"] = player_df["batting_war"] + player_df["pitching_war"]

    # Attach salary
    sal = (
        salaries[salaries["yearID"] >= min_year][["yearID", "teamID", "playerID", "salary"]]
        .rename(
            columns={
                "yearID": "season_key",
                "teamID": "team_id",
                "playerID": "player_id",
            }
        )
        .drop_duplicates(["season_key", "team_id", "player_id"])
    )
    player_df = player_df.merge(sal, on=["player_id", "season_key", "team_id"], how="left")
    player_df["salary"] = player_df["salary"].fillna(0)

    player_df["surplus_value"] = player_surplus_value(player_df["salary"], player_df["player_war"])
    player_df["contract_label"] = classify_contract(player_df["surplus_value"], player_df["player_war"])

    # Coerce numeric columns that may come from outer merge as float
    numeric_cols = ["pa", "woba", "hr", "bb", "ip", "fip", "era"]
    for c in numeric_cols:
        if c not in player_df.columns:
            player_df[c] = np.nan

    # Build output column list dynamically — only include cols that exist
    out_cols = [
        "player_id", "season_key", "team_id", "player_type",
        "pa", "hr", "bb", "woba", "batting_war",
        "ip", "fip", "era", "pitching_war",
        "player_war", "salary", "surplus_value", "contract_label",
    ]
    out_cols = [c for c in out_cols if c in player_df.columns]
    return player_df[out_cols]


# ---------------------------------------------------------------------------
# Fact: team season
# ---------------------------------------------------------------------------

def build_fact_team_season(
    teams: pd.DataFrame,
    salaries: pd.DataFrame,
    batting: pd.DataFrame,
    pitching: pd.DataFrame,
    min_year: int,
) -> pd.DataFrame:
    import numpy as np  # local import so module is importable without numpy side effects

    # ---- salary aggregates ----
    sal_filtered = salaries[salaries["yearID"] >= min_year].copy()
    salary_grouped = (
        sal_filtered
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(
            payroll=("salary", "sum"),
            max_salary=("salary", "max"),
            median_salary=("salary", "median"),
        )
    )

    salary_shares = (
        sal_filtered
        .groupby(["yearID", "teamID"], as_index=False)
        .apply(
            lambda g: pd.concat([top_salary_shares(g), salary_concentration(g)]),
            include_groups=False,
        )
        .reset_index()
    )
    for drop_col in ["level_2", "level_1"]:
        if drop_col in salary_shares.columns:
            salary_shares = salary_shares.drop(columns=[drop_col])

    # ---- WAR aggregates ----
    bat_war_df = batting_war(batting[batting["yearID"] >= min_year])
    pit_war_df = pitching_war(pitching[pitching["yearID"] >= min_year])
    war_team = team_war_totals(bat_war_df, pit_war_df)
    war_team = war_team.rename(columns={"yearID": "yearID_war", "teamID": "teamID_war"})

    # ---- BaseRuns from batting aggregate ----
    bat_team = (
        batting[batting["yearID"] >= min_year]
        .assign(
            _2B=lambda d: d["X2B"] if "X2B" in d.columns else d.get("2B", 0),
            _3B=lambda d: d["X3B"] if "X3B" in d.columns else d.get("3B", 0),
        )
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(
            H=("H", "sum"),
            _2B=("_2B", "sum"),
            _3B=("_3B", "sum"),
            HR=("HR", "sum"),
            BB=("BB", "sum"),
            HBP=("HBP", "sum"),
            AB=("AB", "sum"),
            SF=("SF", "sum"),
        )
    )
    bat_team["HBP"] = bat_team["HBP"].fillna(0)
    bat_team["SF"] = bat_team["SF"].fillna(0)
    bat_team["1B"] = bat_team["H"] - bat_team["_2B"] - bat_team["_3B"] - bat_team["HR"]
    bat_team["base_runs_est"] = base_runs(
        bat_team["H"],
        bat_team["1B"],
        bat_team["_2B"],
        bat_team["_3B"],
        bat_team["HR"],
        bat_team["BB"],
        bat_team["HBP"],
        bat_team["AB"],
        bat_team["SF"],
    )

    # ---- Player-level dead money share ----
    bat_war_merged = bat_war_df.merge(
        sal_filtered[["yearID", "teamID", "playerID", "salary"]],
        on=["yearID", "teamID", "playerID"],
        how="left",
    ).rename(columns={"batting_war": "player_war"})
    pit_war_merged = pit_war_df.merge(
        sal_filtered[["yearID", "teamID", "playerID", "salary"]],
        on=["yearID", "teamID", "playerID"],
        how="left",
    ).rename(columns={"pitching_war": "player_war"})
    all_players = pd.concat([bat_war_merged, pit_war_merged], ignore_index=True)
    all_players["salary"] = all_players["salary"].fillna(0)

    dead_money = (
        all_players
        .groupby(["yearID", "teamID"], as_index=False)
        .apply(
            lambda g: pd.Series({
                "dead_money_share": payroll_underperformer_share(g, "salary", "player_war"),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    for drop_col in ["level_2", "level_1"]:
        if drop_col in dead_money.columns:
            dead_money = dead_money.drop(columns=[drop_col])

    # ---- Merge everything onto teams ----
    t = teams.copy()
    t["run_diff"] = t["R"] - t["RA"]
    t["pythag_wins"] = pythagorean_wins(t["R"], t["RA"], t["G"])

    t = t.merge(salary_grouped, on=["yearID", "teamID"], how="left")
    t = t.merge(salary_shares, on=["yearID", "teamID"], how="left")
    t = t.merge(
        war_team.rename(columns={"yearID_war": "yearID", "teamID_war": "teamID"}),
        on=["yearID", "teamID"],
        how="left",
    )
    t = t.merge(bat_team[["yearID", "teamID", "base_runs_est"]], on=["yearID", "teamID"], how="left")
    t = t.merge(dead_money, on=["yearID", "teamID"], how="left")

    # ---- Derived metrics ----
    t["pythag_gap"] = pythag_gap(t["W"], t["pythag_wins"])
    t["base_runs_gap"] = t["R"] - t["base_runs_est"]
    t["war_win_gap"] = war_win_gap(t["W"], t["team_total_war"].fillna(0))
    t["payroll_per_win"] = safe_divide(t["payroll"], t["W"])
    t["wins_per_10m"] = safe_divide(t["W"] * 10_000_000, t["payroll"])
    t["run_diff_per_10m"] = safe_divide(t["run_diff"] * 10_000_000, t["payroll"])
    t["cost_per_war"] = cost_per_war(t["payroll"], t["team_total_war"])
    t["war_per_1m"] = war_per_dollar(t["team_total_war"], t["payroll"], scale=1_000_000.0)
    t["surplus_value"] = surplus_value_team(t["payroll"], t["team_total_war"].fillna(0))

    # ---- Window detection ----
    window_frames = []
    for team_id, grp in t.groupby("teamID"):
        grp_w = grp.rename(columns={"yearID": "year_id", "W": "wins"}).copy()
        grp_w = detect_team_window(grp_w, win_col="wins", payroll_col="payroll")
        window_frames.append(grp_w[["team_key", "window_phase"]])
    window_df = pd.concat(window_frames, ignore_index=True)
    t = t.merge(window_df, on="team_key", how="left")

    # ---- Rename + select columns ----
    t = t.rename(
        columns={
            "W": "wins",
            "L": "losses",
            "G": "games",
            "R": "runs_scored",
            "RA": "runs_allowed",
            "SOA": "strikeouts",
            "attend": "attendance",
            "base_runs_est": "base_runs",
        }
    )

    cols = [
        "team_key", "season_key",
        "wins", "losses", "games", "runs_scored", "runs_allowed", "strikeouts", "attendance",
        "run_diff", "pythag_wins", "pythag_gap",
        "base_runs", "base_runs_gap",
        "team_batting_war", "team_pitching_war", "team_total_war", "war_win_gap",
        "payroll", "max_salary", "median_salary",
        "top_1_salary_share", "top_3_salary_share", "top_5_salary_share", "gini_salary", "dead_money_share",
        "payroll_per_win", "wins_per_10m", "run_diff_per_10m",
        "cost_per_war", "war_per_1m", "surplus_value",
        "window_phase",
    ]
    # Keep only columns that exist (guard against rare Lahman version differences)
    cols = [c for c in cols if c in t.columns]
    return t[cols]


# ---------------------------------------------------------------------------
# Main pipeline entry
# ---------------------------------------------------------------------------

def build_all(settings: dict) -> tuple[pd.DataFrame, ...]:
    import numpy as np  # noqa: F401 — ensure numpy is available in scope

    min_year = settings["min_year"]
    raw_dir = Path(settings["raw_dir"])

    log.info("Loading raw CSVs from %s", raw_dir)
    teams = read_csv(raw_dir / "teams.csv")
    salaries = read_csv(raw_dir / "salaries.csv")
    people = read_csv(raw_dir / "people.csv")
    batting = read_csv(raw_dir / "batting.csv")
    pitching = read_csv(raw_dir / "pitching.csv")

    teams = teams[
        (teams["yearID"] >= min_year) & (teams["lgID"].isin(["AL", "NL"]))
    ].copy()
    teams["team_key"] = teams["teamID"].astype(str) + "_" + teams["yearID"].astype(str)
    teams["season_key"] = teams["yearID"]

    log.info("Building dimension tables")
    dim_team = build_dim_team(teams)
    dim_season = build_dim_season(teams)
    dim_player = build_dim_player(people)

    log.info("Building fact_salary")
    fact_salary = build_fact_salary(salaries, min_year)

    log.info("Building fact_player_season")
    fact_player_season = build_fact_player_season(batting, pitching, salaries, min_year)

    log.info("Building fact_team_season")
    fact_team_season = build_fact_team_season(teams, salaries, batting, pitching, min_year)

    return dim_team, dim_season, dim_player, fact_salary, fact_player_season, fact_team_season


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    warehouse_path = Path(settings["warehouse_path"])
    ensure_dir(warehouse_path.parent)

    dim_team, dim_season, dim_player, fact_salary, fact_player_season, fact_team_season = build_all(settings)

    log.info("Loading warehouse at %s", warehouse_path)
    con = duckdb.connect(str(warehouse_path))
    con.execute(WAREHOUSE_DDL)

    tables = {
        "dim_team": dim_team,
        "dim_season": dim_season,
        "dim_player": dim_player,
        "fact_salary": fact_salary,
        "fact_player_season": fact_player_season,
        "fact_team_season": fact_team_season,
    }
    for table_name, df in tables.items():
        # DDL used CREATE OR REPLACE so the table is always fresh — no DELETE needed.
        # Use explicit column list to be safe against column order mismatches.
        # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
        db_cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table_name}')").fetchall()]
        common = [c for c in df.columns if c in db_cols]
        if not common:
            log.error(
                "No overlapping columns between DataFrame (%s) and table %s (%s). Skipping.",
                list(df.columns),
                table_name,
                db_cols,
            )
            continue
        view_name = f"_load_{table_name}"
        con.register(view_name, df[common])
        col_list = ", ".join(common)
        con.execute(f"INSERT INTO {table_name} ({col_list}) SELECT {col_list} FROM {view_name}")
        con.unregister(view_name)
        log.info("Loaded %d rows into %s", len(df), table_name)

    con.close()

    # ---- Post-load validation ----
    from src.baseball_analytics.validation import validate_all
    report = validate_all(fact_team_season, fact_player_season, dim_team)
    if not report.passed:
        log.warning("Validation issues detected: %s", report.summary())
    else:
        log.info("Validation passed: %s", report.summary())

    typer.echo(f"Warehouse built: {warehouse_path}")


if __name__ == "__main__":
    app()
