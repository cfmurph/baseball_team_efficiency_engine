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
-- One row per player per season.
-- Players who were traded mid-season have their stats summed across teams;
-- team_name shows the team where they accrued the most WAR.
SELECT
    p.player_id,
    dp.name_full,
    dp.name_first,
    dp.name_last,
    p.season_key                        AS year_id,

    -- Primary team = team with highest WAR that season
    FIRST(p.team_id ORDER BY p.player_war DESC NULLS LAST)  AS team_id,
    FIRST(t.team_name ORDER BY p.player_war DESC NULLS LAST) AS team_name,

    -- Pick the most specific player type (both > pitcher > batter)
    CASE
        WHEN SUM(CASE WHEN p.player_type = 'both'    THEN 1 ELSE 0 END) > 0 THEN 'both'
        WHEN SUM(CASE WHEN p.player_type = 'pitcher' THEN 1 ELSE 0 END) > 0 THEN 'pitcher'
        ELSE 'batter'
    END                                 AS player_type,

    SUM(p.pa)                           AS pa,
    SUM(p.hr)                           AS hr,
    SUM(p.bb)                           AS bb,
    AVG(CASE WHEN p.pa > 0 THEN p.woba END) AS woba,

    SUM(p.batting_war)                  AS batting_war,
    SUM(p.ip)                           AS ip,
    AVG(CASE WHEN p.ip > 0 THEN p.fip END) AS fip,
    AVG(CASE WHEN p.ip > 0 THEN p.era END) AS era,
    SUM(p.pitching_war)                 AS pitching_war,
    SUM(p.player_war)                   AS player_war,

    SUM(p.salary)                       AS salary,
    SUM(p.surplus_value)                AS surplus_value,

    -- Contract label from the stint with the most WAR
    FIRST(p.contract_label ORDER BY p.player_war DESC NULLS LAST) AS contract_label

FROM fact_player_season p
LEFT JOIN dim_player dp USING (player_id)
LEFT JOIN (
    SELECT DISTINCT team_id, team_name
    FROM dim_team
) t ON t.team_id = p.team_id
GROUP BY p.player_id, dp.name_full, dp.name_first, dp.name_last, p.season_key
ORDER BY p.season_key, SUM(p.player_war) DESC
"""

# Sportradar enrichment: real WAR + wOBA + wRC+ + FIP/ERA-
# Only available for seasons/players that have been pulled via pull_sportradar.py
_SR_PLAYER_QUERY = """
SELECT
    sp.sr_player_id,
    sp.full_name,
    sp.season_year  AS year_id,
    sp.sr_team_id,
    m.lahman_team_id AS team_id,
    sp.primary_position,
    sp.pa,
    sp.hr,
    sp.woba,
    sp.wrc_plus,
    sp.war,
    sp.bwar,
    sp.fwar,
    sp.p_war,
    COALESCE(sp.war, sp.p_war, 0) AS player_war_sr,
    sp.ip,
    sp.era,
    sp.era_minus,
    sp.fip,
    sp.k9,
    sp.bb9,
    sp.kbb
FROM fact_sr_player_season sp
LEFT JOIN dim_sportradar_team_map m USING (sr_team_id)
ORDER BY sp.season_year, player_war_sr DESC
"""

_SR_TRANSACTIONS_QUERY = """
SELECT
    t.transaction_id,
    t.effective_date,
    t.transaction_type,
    t.transaction_code,
    t.description,
    t.player_name,
    t.from_team_abbr,
    t.to_team_abbr
FROM fact_sr_transactions t
ORDER BY t.effective_date DESC
"""

_SR_INJURIES_QUERY = """
SELECT
    i.sr_player_id,
    i.player_name,
    i.team_abbr,
    i.injury_desc,
    i.injury_status,
    i.start_date,
    i.end_date
FROM fact_sr_injuries i
ORDER BY i.start_date DESC
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


def _table_has_rows(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    """Return True if the table exists and has at least one row."""
    try:
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return n > 0
    except Exception:
        return False


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

    # ---- Sportradar enrichment (only if data was pulled) ----
    sr_player_df: pd.DataFrame | None = None
    sr_tx_df: pd.DataFrame | None = None
    sr_injury_df: pd.DataFrame | None = None

    if _table_has_rows(con, "fact_sr_player_season"):
        log.info("Sportradar player stats available — exporting")
        sr_player_df = con.execute(_SR_PLAYER_QUERY).fetchdf()
    else:
        log.info("No Sportradar player stats found (run pull_sportradar.py to add them)")

    if _table_has_rows(con, "fact_sr_transactions"):
        sr_tx_df = con.execute(_SR_TRANSACTIONS_QUERY).fetchdf()
        log.info("Sportradar transactions: %d rows", len(sr_tx_df))

    if _table_has_rows(con, "fact_sr_injuries"):
        sr_injury_df = con.execute(_SR_INJURIES_QUERY).fetchdf()
        log.info("Sportradar injuries: %d rows", len(sr_injury_df))

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

    # ---- Sportradar exports (only if data present) ----
    if sr_player_df is not None:
        sr_player_df.to_csv(artifacts_dir / "sr_player_season_metrics.csv", index=False)
        log.info("Wrote sr_player_season_metrics.csv (%d rows)", len(sr_player_df))

        # WAR leaderboard — real values from Sportradar
        war_leaders = (
            sr_player_df[sr_player_df["player_war_sr"] > 0]
            .nlargest(200, "player_war_sr")
            .reset_index(drop=True)
        )
        war_leaders.to_csv(artifacts_dir / "sr_war_leaders.csv", index=False)

        # wRC+ leaders (quality of contact)
        if "wrc_plus" in sr_player_df.columns:
            wrc_leaders = (
                sr_player_df[sr_player_df["wrc_plus"].notna() & (sr_player_df["pa"] >= 100)]
                .nlargest(100, "wrc_plus")
                .reset_index(drop=True)
            )
            wrc_leaders.to_csv(artifacts_dir / "sr_wrc_plus_leaders.csv", index=False)

        # ERA- leaders (pitching quality)
        if "era_minus" in sr_player_df.columns:
            era_minus_leaders = (
                sr_player_df[sr_player_df["era_minus"].notna() & (sr_player_df["ip"] >= 20)]
                .nsmallest(100, "era_minus")
                .reset_index(drop=True)
            )
            era_minus_leaders.to_csv(artifacts_dir / "sr_era_minus_leaders.csv", index=False)

    if sr_tx_df is not None:
        sr_tx_df.to_csv(artifacts_dir / "sr_transactions.csv", index=False)
        log.info("Wrote sr_transactions.csv (%d rows)", len(sr_tx_df))

    if sr_injury_df is not None:
        sr_injury_df.to_csv(artifacts_dir / "sr_injuries.csv", index=False)
        log.info("Wrote sr_injuries.csv (%d rows)", len(sr_injury_df))

    typer.echo(f"Wrote all artifacts to {artifacts_dir}")


if __name__ == "__main__":
    app()
