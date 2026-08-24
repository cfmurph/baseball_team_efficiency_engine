from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.fantasy import emit_ranked_fantasy_cards
from src.baseball_analytics.io import ensure_dir
from src.baseball_analytics.metrics import pythag_gap, pythagorean_wins, war_win_gap
from src.baseball_analytics.sportsdataio import (
    ENDPOINT_EXTRACT_REPORT,
    default_season_window,
    extract_had_in_season,
    read_raw_payload,
    seasons_from_settings,
)
from src.baseball_analytics.storage import default_as_of_date

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
    f.war_source,
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
-- rate stats are weighted by playing time, and team_name shows the team
-- where they accrued the most WAR.
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
    SUM(CASE WHEN p.pa > 0 AND p.woba IS NOT NULL THEN p.woba * p.pa END)
        / NULLIF(SUM(CASE WHEN p.pa > 0 AND p.woba IS NOT NULL THEN p.pa ELSE 0 END), 0) AS woba,

    SUM(p.batting_war)                  AS batting_war,
    SUM(p.ip)                           AS ip,
    SUM(CASE WHEN p.ip > 0 AND p.fip IS NOT NULL THEN p.fip * p.ip END)
        / NULLIF(SUM(CASE WHEN p.ip > 0 AND p.fip IS NOT NULL THEN p.ip ELSE 0 END), 0) AS fip,
    SUM(CASE WHEN p.ip > 0 AND p.era IS NOT NULL THEN p.era * p.ip END)
        / NULLIF(SUM(CASE WHEN p.ip > 0 AND p.era IS NOT NULL THEN p.ip ELSE 0 END), 0) AS era,
    SUM(p.pitching_war)                 AS pitching_war,
    SUM(p.player_war)                   AS player_war,
    CASE
        WHEN COUNT(DISTINCT p.war_source) = 1 THEN MIN(p.war_source)
        ELSE 'mixed'
    END                                 AS war_source,

    SUM(p.salary)                       AS salary,
    SUM(p.surplus_value)                AS surplus_value,

    -- Contract label from the stint with the most WAR
    FIRST(p.contract_label ORDER BY p.player_war DESC NULLS LAST) AS contract_label

FROM fact_player_season p
LEFT JOIN dim_player dp USING (player_id)
LEFT JOIN dim_team t
    ON t.team_key = p.team_id || '_' || CAST(p.season_key AS VARCHAR)
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

# SportsDataIO live season overlay. Does not read or write WAR onto spine facts.
# Prefers a Lahman playerID alias when one exists so historical rows stay joined.
_SDIO_PLAYER_SEASON_QUERY = """
SELECT
    COALESCE(lahman.external_id, s.player_id) AS player_id,
    COALESCE(p.display_name, s.player_id)     AS name_full,
    p.first_name                              AS name_first,
    p.last_name                               AS name_last,
    s.season                                  AS year_id,
    FIRST(COALESCE(t.sdio_abbr, t.team_id) ORDER BY COALESCE(s.pa, 0) + COALESCE(s.ip, 0) DESC) AS team_id,
    FIRST(COALESCE(t.team_name, t.sdio_abbr) ORDER BY COALESCE(s.pa, 0) + COALESCE(s.ip, 0) DESC) AS team_name,
    CASE
        WHEN SUM(COALESCE(s.ip, 0)) > 0 AND SUM(COALESCE(s.pa, 0)) > 0 THEN 'both'
        WHEN SUM(COALESCE(s.ip, 0)) > 0 THEN 'pitcher'
        ELSE 'batter'
    END                                       AS player_type,
    FIRST(p.position ORDER BY COALESCE(s.pa, 0) + COALESCE(s.ip, 0) DESC) AS position,
    SUM(s.pa)                                 AS pa,
    SUM(s.hr)                                 AS hr,
    SUM(s.bb)                                 AS bb,
    SUM(s.hits)                               AS hits,
    SUM(s.games)                              AS games,
    SUM(s.ab)                                 AS ab,
    SUM(s.so)                                 AS so,
    SUM(s.rbi)                                AS rbi,
    SUM(s.sb)                                 AS sb,
    CAST(NULL AS DOUBLE)                      AS woba,
    CAST(NULL AS DOUBLE)                      AS batting_war,
    SUM(s.ip)                                 AS ip,
    CAST(NULL AS DOUBLE)                      AS fip,
    SUM(CASE WHEN s.ip > 0 AND s.era IS NOT NULL THEN s.era * s.ip END)
        / NULLIF(SUM(CASE WHEN s.ip > 0 AND s.era IS NOT NULL THEN s.ip ELSE 0 END), 0) AS era,
    SUM(CASE WHEN s.ip > 0 AND s.whip IS NOT NULL THEN s.whip * s.ip END)
        / NULLIF(SUM(CASE WHEN s.ip > 0 AND s.whip IS NOT NULL THEN s.ip ELSE 0 END), 0) AS whip,
    SUM(s.pitching_so)                        AS pitching_so,
    SUM(s.pitching_bb)                        AS pitching_bb,
    CAST(NULL AS DOUBLE)                      AS pitching_war,
    CAST(NULL AS DOUBLE)                      AS player_war,
    'approx'                                  AS war_source,
    CAST(NULL AS DOUBLE)                      AS salary,
    CAST(NULL AS DOUBLE)                      AS surplus_value,
    CAST(NULL AS VARCHAR)                     AS contract_label,
    'sportsdataio'                            AS stat_source
FROM player_season_stat s
LEFT JOIN player p ON p.player_id = s.player_id
LEFT JOIN team t ON t.team_id = s.team_id
LEFT JOIN (
    SELECT internal_id, MIN(external_id) AS external_id
    FROM external_id_alias
    WHERE system = 'lahman' AND entity_type = 'player'
    GROUP BY internal_id
) lahman ON lahman.internal_id = s.player_id
GROUP BY
    COALESCE(lahman.external_id, s.player_id),
    COALESCE(p.display_name, s.player_id),
    p.first_name,
    p.last_name,
    s.season
ORDER BY s.season, SUM(COALESCE(s.pa, 0)) DESC
"""

_SDIO_PLAYER_GAME_ROLLUP_QUERY = """
SELECT
    COALESCE(lahman.external_id, g.player_id) AS player_id,
    COALESCE(p.display_name, g.player_id)     AS name_full,
    p.first_name                              AS name_first,
    p.last_name                               AS name_last,
    g.season                                  AS year_id,
    FIRST(COALESCE(t.sdio_abbr, t.team_id) ORDER BY COALESCE(g.pa, 0) + COALESCE(g.ip, 0) DESC) AS team_id,
    FIRST(COALESCE(t.team_name, t.sdio_abbr) ORDER BY COALESCE(g.pa, 0) + COALESCE(g.ip, 0) DESC) AS team_name,
    CASE
        WHEN SUM(COALESCE(g.ip, 0)) > 0 AND SUM(COALESCE(g.pa, 0)) > 0 THEN 'both'
        WHEN SUM(COALESCE(g.ip, 0)) > 0 THEN 'pitcher'
        ELSE 'batter'
    END                                       AS player_type,
    FIRST(COALESCE(g.position, p.position) ORDER BY COALESCE(g.pa, 0) + COALESCE(g.ip, 0) DESC) AS position,
    SUM(g.pa)                                 AS pa,
    SUM(g.hr)                                 AS hr,
    SUM(g.bb)                                 AS bb,
    SUM(g.hits)                               AS hits,
    COUNT(*)                                  AS games,
    SUM(g.ab)                                 AS ab,
    SUM(g.so)                                 AS so,
    SUM(g.rbi)                                AS rbi,
    SUM(g.sb)                                 AS sb,
    CAST(NULL AS DOUBLE)                      AS woba,
    CAST(NULL AS DOUBLE)                      AS batting_war,
    SUM(g.ip)                                 AS ip,
    CAST(NULL AS DOUBLE)                      AS fip,
    SUM(CASE WHEN g.ip > 0 AND g.era IS NOT NULL THEN g.era * g.ip END)
        / NULLIF(SUM(CASE WHEN g.ip > 0 AND g.era IS NOT NULL THEN g.ip ELSE 0 END), 0) AS era,
    SUM(CASE WHEN g.ip > 0 AND g.whip IS NOT NULL THEN g.whip * g.ip END)
        / NULLIF(SUM(CASE WHEN g.ip > 0 AND g.whip IS NOT NULL THEN g.ip ELSE 0 END), 0) AS whip,
    SUM(g.pitching_so)                        AS pitching_so,
    SUM(g.pitching_bb)                        AS pitching_bb,
    CAST(NULL AS DOUBLE)                      AS pitching_war,
    CAST(NULL AS DOUBLE)                      AS player_war,
    'approx'                                  AS war_source,
    CAST(NULL AS DOUBLE)                      AS salary,
    CAST(NULL AS DOUBLE)                      AS surplus_value,
    CAST(NULL AS VARCHAR)                     AS contract_label,
    'sportsdataio'                            AS stat_source
FROM player_game_stat g
LEFT JOIN player p ON p.player_id = g.player_id
LEFT JOIN team t ON t.team_id = g.team_id
LEFT JOIN (
    SELECT internal_id, MIN(external_id) AS external_id
    FROM external_id_alias
    WHERE system = 'lahman' AND entity_type = 'player'
    GROUP BY internal_id
) lahman ON lahman.internal_id = g.player_id
GROUP BY
    COALESCE(lahman.external_id, g.player_id),
    COALESCE(p.display_name, g.player_id),
    p.first_name,
    p.last_name,
    g.season
ORDER BY g.season, SUM(COALESCE(g.pa, 0)) DESC
"""

# Team grain of the #133 overlay. v0.1 has no team_season_stat — roll up
# player_season_stat / player_game_stat by team_id + season. Do not write WAR
# onto spine facts. Payroll/salary stay null (do not invent).
_SDIO_TEAM_SEASON_QUERY = """
SELECT
    COALESCE(lahman.external_id, t.sdio_abbr, s.team_id) AS team_id,
    FIRST(
        COALESCE(
            NULLIF(TRIM(COALESCE(t.city, '') || ' ' || COALESCE(t.team_name, '')), ''),
            t.team_name,
            t.sdio_abbr,
            CAST(s.team_id AS VARCHAR)
        )
        ORDER BY COALESCE(s.pa, 0) + COALESCE(s.ip, 0) DESC
    ) AS team_name,
    FIRST(COALESCE(lahman.external_id, t.sdio_abbr)) AS franchise_id,
    FIRST(t.league) AS league_id,
    s.season AS year_id,
    CAST(NULL AS BIGINT) AS wins,
    CAST(NULL AS BIGINT) AS losses,
    MAX(s.games) AS games,
    CAST(NULL AS DOUBLE) AS runs_scored,
    CAST(NULL AS DOUBLE) AS runs_allowed,
    SUM(s.pa) AS pa,
    SUM(s.hr) AS hr,
    SUM(s.bb) AS bb,
    SUM(s.hits) AS hits,
    SUM(s.ab) AS ab,
    SUM(s.so) AS so,
    SUM(s.rbi) AS rbi,
    SUM(s.sb) AS sb,
    SUM(s.ip) AS ip,
    SUM(s.pitching_so) AS pitching_so,
    SUM(s.pitching_bb) AS pitching_bb,
    CAST(NULL AS DOUBLE) AS team_batting_war,
    CAST(NULL AS DOUBLE) AS team_pitching_war,
    CAST(NULL AS DOUBLE) AS team_total_war,
    'approx' AS war_source,
    CAST(NULL AS DOUBLE) AS payroll,
    CAST(NULL AS DOUBLE) AS max_salary,
    CAST(NULL AS DOUBLE) AS median_salary,
    CAST(NULL AS DOUBLE) AS top_1_salary_share,
    CAST(NULL AS DOUBLE) AS top_3_salary_share,
    CAST(NULL AS DOUBLE) AS top_5_salary_share,
    CAST(NULL AS DOUBLE) AS gini_salary,
    CAST(NULL AS DOUBLE) AS dead_money_share,
    CAST(NULL AS DOUBLE) AS payroll_per_win,
    CAST(NULL AS DOUBLE) AS wins_per_10m,
    CAST(NULL AS DOUBLE) AS run_diff_per_10m,
    CAST(NULL AS DOUBLE) AS cost_per_war,
    CAST(NULL AS DOUBLE) AS war_per_1m,
    CAST(NULL AS DOUBLE) AS surplus_value,
    CAST(NULL AS VARCHAR) AS window_phase,
    'sportsdataio' AS stat_source
FROM player_season_stat s
LEFT JOIN team t ON t.team_id = s.team_id
LEFT JOIN (
    SELECT internal_id, MIN(external_id) AS external_id
    FROM external_id_alias
    WHERE system = 'lahman' AND entity_type = 'team'
    GROUP BY internal_id
) lahman ON lahman.internal_id = s.team_id
WHERE s.team_id IS NOT NULL
GROUP BY COALESCE(lahman.external_id, t.sdio_abbr, s.team_id), s.season
ORDER BY s.season, SUM(COALESCE(s.pa, 0)) DESC
"""

_SDIO_TEAM_GAME_ROLLUP_QUERY = """
SELECT
    COALESCE(lahman.external_id, t.sdio_abbr, g.team_id) AS team_id,
    FIRST(
        COALESCE(
            NULLIF(TRIM(COALESCE(t.city, '') || ' ' || COALESCE(t.team_name, '')), ''),
            t.team_name,
            t.sdio_abbr,
            CAST(g.team_id AS VARCHAR)
        )
        ORDER BY COALESCE(g.pa, 0) + COALESCE(g.ip, 0) DESC
    ) AS team_name,
    FIRST(COALESCE(lahman.external_id, t.sdio_abbr)) AS franchise_id,
    FIRST(t.league) AS league_id,
    g.season AS year_id,
    CAST(NULL AS BIGINT) AS wins,
    CAST(NULL AS BIGINT) AS losses,
    COUNT(DISTINCT g.game_id) AS games,
    SUM(g.runs) AS runs_scored,
    CAST(NULL AS DOUBLE) AS runs_allowed,
    SUM(g.pa) AS pa,
    SUM(g.hr) AS hr,
    SUM(g.bb) AS bb,
    SUM(g.hits) AS hits,
    SUM(g.ab) AS ab,
    SUM(g.so) AS so,
    SUM(g.rbi) AS rbi,
    SUM(g.sb) AS sb,
    SUM(g.ip) AS ip,
    SUM(g.pitching_so) AS pitching_so,
    SUM(g.pitching_bb) AS pitching_bb,
    CAST(NULL AS DOUBLE) AS team_batting_war,
    CAST(NULL AS DOUBLE) AS team_pitching_war,
    CAST(NULL AS DOUBLE) AS team_total_war,
    'approx' AS war_source,
    CAST(NULL AS DOUBLE) AS payroll,
    CAST(NULL AS DOUBLE) AS max_salary,
    CAST(NULL AS DOUBLE) AS median_salary,
    CAST(NULL AS DOUBLE) AS top_1_salary_share,
    CAST(NULL AS DOUBLE) AS top_3_salary_share,
    CAST(NULL AS DOUBLE) AS top_5_salary_share,
    CAST(NULL AS DOUBLE) AS gini_salary,
    CAST(NULL AS DOUBLE) AS dead_money_share,
    CAST(NULL AS DOUBLE) AS payroll_per_win,
    CAST(NULL AS DOUBLE) AS wins_per_10m,
    CAST(NULL AS DOUBLE) AS run_diff_per_10m,
    CAST(NULL AS DOUBLE) AS cost_per_war,
    CAST(NULL AS DOUBLE) AS war_per_1m,
    CAST(NULL AS DOUBLE) AS surplus_value,
    CAST(NULL AS VARCHAR) AS window_phase,
    'sportsdataio' AS stat_source
FROM player_game_stat g
LEFT JOIN team t ON t.team_id = g.team_id
LEFT JOIN (
    SELECT internal_id, MIN(external_id) AS external_id
    FROM external_id_alias
    WHERE system = 'lahman' AND entity_type = 'team'
    GROUP BY internal_id
) lahman ON lahman.internal_id = g.team_id
WHERE g.team_id IS NOT NULL
GROUP BY COALESCE(lahman.external_id, t.sdio_abbr, g.team_id), g.season
ORDER BY g.season, SUM(COALESCE(g.pa, 0)) DESC
"""

_SDIO_TEAM_STANDINGS_QUERY = """
WITH team_games AS (
    SELECT
        home_team_id AS team_id,
        season,
        home_score AS team_score,
        away_score AS opp_score
    FROM game
    WHERE home_score IS NOT NULL AND away_score IS NOT NULL
    UNION ALL
    SELECT
        away_team_id AS team_id,
        season,
        away_score AS team_score,
        home_score AS opp_score
    FROM game
    WHERE home_score IS NOT NULL AND away_score IS NOT NULL
)
SELECT
    COALESCE(lahman.external_id, t.sdio_abbr, g.team_id) AS team_id,
    FIRST(
        COALESCE(
            NULLIF(TRIM(COALESCE(t.city, '') || ' ' || COALESCE(t.team_name, '')), ''),
            t.team_name,
            t.sdio_abbr,
            CAST(g.team_id AS VARCHAR)
        )
    ) AS team_name,
    FIRST(COALESCE(lahman.external_id, t.sdio_abbr)) AS franchise_id,
    FIRST(t.league) AS league_id,
    g.season AS year_id,
    SUM(CASE WHEN g.team_score > g.opp_score THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN g.team_score < g.opp_score THEN 1 ELSE 0 END) AS losses,
    COUNT(*) AS games,
    SUM(g.team_score) AS runs_scored,
    SUM(g.opp_score) AS runs_allowed,
    CAST(0 AS DOUBLE) AS pa,
    CAST(0 AS DOUBLE) AS hr,
    CAST(0 AS DOUBLE) AS bb,
    CAST(0 AS DOUBLE) AS hits,
    CAST(0 AS DOUBLE) AS ab,
    CAST(0 AS DOUBLE) AS so,
    CAST(0 AS DOUBLE) AS rbi,
    CAST(0 AS DOUBLE) AS sb,
    CAST(0 AS DOUBLE) AS ip,
    CAST(0 AS DOUBLE) AS pitching_so,
    CAST(0 AS DOUBLE) AS pitching_bb,
    CAST(NULL AS DOUBLE) AS team_batting_war,
    CAST(NULL AS DOUBLE) AS team_pitching_war,
    CAST(NULL AS DOUBLE) AS team_total_war,
    'approx' AS war_source,
    CAST(NULL AS DOUBLE) AS payroll,
    CAST(NULL AS VARCHAR) AS window_phase,
    'sportsdataio' AS stat_source
FROM team_games g
LEFT JOIN team t ON t.team_id = g.team_id
LEFT JOIN (
    SELECT internal_id, MIN(external_id) AS external_id
    FROM external_id_alias
    WHERE system = 'lahman' AND entity_type = 'team'
    GROUP BY internal_id
) lahman ON lahman.internal_id = g.team_id
WHERE g.team_id IS NOT NULL
GROUP BY COALESCE(lahman.external_id, t.sdio_abbr, g.team_id), g.season
ORDER BY g.season
"""

METRICS_MANIFEST_NAME = "metrics_manifest.json"

TEAM_PUBLISH_COLUMNS = (
    "year_id",
    "team_name",
    "team_id",
    "franchise_id",
    "league_id",
    "wins",
    "losses",
    "games",
    "runs_scored",
    "runs_allowed",
    "run_diff",
    "pythag_wins",
    "pythag_gap",
    "base_runs",
    "base_runs_gap",
    "team_batting_war",
    "team_pitching_war",
    "team_total_war",
    "war_source",
    "war_win_gap",
    "payroll",
    "max_salary",
    "median_salary",
    "top_1_salary_share",
    "top_3_salary_share",
    "top_5_salary_share",
    "gini_salary",
    "dead_money_share",
    "payroll_per_win",
    "wins_per_10m",
    "run_diff_per_10m",
    "cost_per_war",
    "war_per_1m",
    "surplus_value",
    "window_phase",
)


def _efficiency_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wins_per_10m" not in df.columns:
        df["wins_per_10m"] = pd.NA
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


_POSITION_FROM_TYPE = {
    "pitcher": "P",
    "batter": "UTIL",
    "both": "UTIL",
}

PHASE0_PLAYER_FIELDS = (
    "player_id",
    "player_name",
    "team",
    "position",
    "player_war",
    "war",
    "war_source",
    "surplus_value",
    "cost_per_war",
    "vs_replacement",
    "edge",
    "rank_overall",
    "rank_at_position",
    "season",
    "as_of_date",
)


def enrich_player_season_phase0(
    player_df: pd.DataFrame,
    *,
    as_of_date: str,
) -> pd.DataFrame:
    """Add fantasy Phase 0 aliases/ranks without changing player-season grain.

    Existing dashboard columns (``name_full``, ``team_name``, ``year_id``,
    ``player_war``, ``surplus_value``) stay in place. New fields are additive.
    ``war`` aliases ``player_war`` (same value — not a second WAR write).
    """
    out = player_df.copy()
    if out.empty:
        for col in PHASE0_PLAYER_FIELDS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    if "player_name" not in out.columns:
        source = "name_full" if "name_full" in out.columns else None
        out["player_name"] = out[source] if source else pd.NA
    if "team" not in out.columns:
        source = "team_name" if "team_name" in out.columns else None
        out["team"] = out[source] if source else pd.NA
    if "season" not in out.columns:
        if "year_id" in out.columns:
            out["season"] = out["year_id"]
        elif "season_key" in out.columns:
            out["season"] = out["season_key"]
        else:
            out["season"] = pd.NA
    if "as_of_date" not in out.columns:
        out["as_of_date"] = as_of_date
    if "position" not in out.columns:
        if "player_type" in out.columns:
            out["position"] = out["player_type"].map(_POSITION_FROM_TYPE).fillna("UTIL")
        else:
            out["position"] = "UTIL"
    if "war" not in out.columns and "player_war" in out.columns:
        out["war"] = out["player_war"]
    if "cost_per_war" not in out.columns and {"salary", "player_war"} <= set(out.columns):
        war = pd.to_numeric(out["player_war"], errors="coerce")
        salary = pd.to_numeric(out["salary"], errors="coerce")
        out["cost_per_war"] = salary.where(war.abs() > 1e-9) / war.replace(0, pd.NA)
    if "vs_replacement" not in out.columns and "player_war" in out.columns:
        out["vs_replacement"] = out["player_war"]
    if "edge" not in out.columns and "surplus_value" in out.columns:
        out["edge"] = out["surplus_value"]
    if "is_approx" not in out.columns and "war_source" in out.columns:
        out["is_approx"] = ~out["war_source"].astype(str).str.lower().isin({"real", "bbref"})

    season_key = "season" if "season" in out.columns else None
    if "rank_overall" not in out.columns and "player_war" in out.columns and season_key:
        out["rank_overall"] = (
            out.groupby(season_key, dropna=False)["player_war"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
    if (
        "rank_at_position" not in out.columns
        and "player_war" in out.columns
        and season_key
        and "position" in out.columns
    ):
        out["rank_at_position"] = (
            out.groupby([season_key, "position"], dropna=False)["player_war"]
            .rank(method="min", ascending=False)
            .astype("Int64")
        )
    return out


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


@dataclass
class SeasonCoverage:
    """Publish-time signal so prior-only metrics are not mistaken for current."""

    as_of_date: str
    active_season: int
    season_window: list[int]
    seasons_present: list[int]
    overlay_seasons: list[int]
    overlay_rows: int
    active_season_present: bool
    active_season_source: str | None
    current_season_missing: bool
    current_season_missing_reason: str | None
    sdio_in_season: bool = False
    team_seasons_present: list[int] = field(default_factory=list)
    team_overlay_seasons: list[int] = field(default_factory=list)
    team_overlay_rows: int = 0
    team_active_season_present: bool = False
    team_active_season_source: str | None = None
    team_current_season_missing: bool = True
    team_current_season_missing_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SdioPublishFrames:
    """SDIO frames already shaped for the player and team metric overlays."""

    player_season: pd.DataFrame | None = None
    player_game: pd.DataFrame | None = None
    team_season: pd.DataFrame | None = None
    team_game: pd.DataFrame | None = None
    team_standings: pd.DataFrame | None = None


def _year_set(frame: pd.DataFrame, column: str = "year_id") -> set[int]:
    if frame is None or frame.empty or column not in frame.columns:
        return set()
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return {int(year) for year in values}


def _season_frame_is_thin(frame: pd.DataFrame | None, year: int) -> bool:
    """True when the season stub has no counting-stat volume for ``year``."""
    if frame is None or frame.empty or "year_id" not in frame.columns:
        return True
    years = pd.to_numeric(frame["year_id"], errors="coerce")
    subset = frame.loc[years == year]
    if subset.empty:
        return True
    pa = pd.to_numeric(subset["pa"], errors="coerce").fillna(0).sum() if "pa" in subset.columns else 0
    ip = pd.to_numeric(subset["ip"], errors="coerce").fillna(0).sum() if "ip" in subset.columns else 0
    return float(pa) <= 0 and float(ip) <= 0


def approx_vs_replacement_from_counting(row: pd.Series) -> float:
    """Ranking proxy for SDIO overlay rows. Not rWAR and never written to spine facts."""
    pa = float(pd.to_numeric(row.get("pa"), errors="coerce") or 0)
    ip = float(pd.to_numeric(row.get("ip"), errors="coerce") or 0)
    hr = float(pd.to_numeric(row.get("hr"), errors="coerce") or 0)
    hits = float(pd.to_numeric(row.get("hits"), errors="coerce") or 0)
    bb = float(pd.to_numeric(row.get("bb"), errors="coerce") or 0)
    sb = float(pd.to_numeric(row.get("sb"), errors="coerce") or 0)
    so = float(pd.to_numeric(row.get("so"), errors="coerce") or 0)
    rbi = float(pd.to_numeric(row.get("rbi"), errors="coerce") or 0)
    pitching_so = float(pd.to_numeric(row.get("pitching_so"), errors="coerce") or 0)
    pitching_bb = float(pd.to_numeric(row.get("pitching_bb"), errors="coerce") or 0)
    era = pd.to_numeric(row.get("era"), errors="coerce")

    batting = 0.0
    if pa > 0:
        singles = max(hits - hr, 0.0)
        batting = (hr * 1.4 + singles * 0.3 + bb * 0.3 + sb * 0.2 - so * 0.1 + rbi * 0.05) / 10.0
    pitching = 0.0
    if ip > 0:
        era_val = float(era) if pd.notna(era) else 4.5
        pitching = (ip / 50.0) + (pitching_so - pitching_bb) / 40.0 - max(era_val - 4.0, -2.0) * (ip / 180.0)
    return round(batting + pitching, 3)


def _apply_counting_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    if "player_war" not in out.columns:
        out["player_war"] = pd.NA
    missing = pd.to_numeric(out["player_war"], errors="coerce").isna()
    if missing.any():
        out.loc[missing, "player_war"] = out.loc[missing].apply(
            approx_vs_replacement_from_counting, axis=1
        )
        if "batting_war" in out.columns:
            bat_missing = missing & pd.to_numeric(out["batting_war"], errors="coerce").isna()
            out.loc[bat_missing, "batting_war"] = out.loc[bat_missing, "player_war"]
        if "pitching_war" in out.columns:
            pit_missing = missing & pd.to_numeric(out["pitching_war"], errors="coerce").isna()
            pitcherish = out["player_type"].astype(str).isin({"pitcher", "both"}) if "player_type" in out.columns else False
            out.loc[pit_missing & pitcherish, "pitching_war"] = out.loc[pit_missing & pitcherish, "player_war"]
    if "war_source" not in out.columns:
        out["war_source"] = "approx"
    else:
        out["war_source"] = out["war_source"].fillna("approx")
    return out


def select_sdio_overlay_frame(
    season_df: pd.DataFrame | None,
    game_df: pd.DataFrame | None,
    *,
    window: list[int],
    lahman_years: set[int],
) -> pd.DataFrame:
    """Keep SDIO rows for window years Lahman lacks. Roll up games when the stub is thin."""
    need = [year for year in window if year not in lahman_years]
    if not need:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    covered: set[int] = set()
    if season_df is not None and not season_df.empty and "year_id" in season_df.columns:
        years = pd.to_numeric(season_df["year_id"], errors="coerce")
        for year in need:
            if _season_frame_is_thin(season_df, year):
                continue
            slice_df = season_df.loc[years == year]
            if not slice_df.empty:
                parts.append(slice_df)
                covered.add(year)
    still_need = [year for year in need if year not in covered]
    if still_need and game_df is not None and not game_df.empty and "year_id" in game_df.columns:
        years = pd.to_numeric(game_df["year_id"], errors="coerce")
        slice_df = game_df.loc[years.isin(still_need)]
        if not slice_df.empty:
            parts.append(slice_df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _source_frames_present(*frames: pd.DataFrame | None) -> bool:
    return any(frame is not None and not frame.empty for frame in frames)


def coverage_from_frames(
    *,
    as_of_date: str,
    window: list[int],
    combined: pd.DataFrame,
    overlay: pd.DataFrame,
    lahman_years: set[int],
    source_present: bool,
    extract_report: dict | None = None,
    season_df: pd.DataFrame | None = None,
    game_df: pd.DataFrame | None = None,
) -> SeasonCoverage:
    """Shared #131/#138 coverage block. Same window; grain is the combined frame."""
    resolved_window = list(window)
    active = resolved_window[-1] if resolved_window else int(str(as_of_date)[:4])
    overlay_years = sorted(_year_set(overlay))
    seasons_present = sorted(_year_set(combined))
    active_present = active in seasons_present
    if active in overlay_years:
        active_source: str | None = "sportsdataio"
        missing_reason = None
    elif active in lahman_years:
        active_source = "lahman"
        missing_reason = None
    elif source_present:
        active_source = None
        missing_reason = "sdio_empty_active_season"
    else:
        active_source = None
        missing_reason = "sdio_unavailable"
    sdio_in_season = extract_had_in_season(extract_report, active_season=active)
    if not sdio_in_season:
        sdio_in_season = active in _year_set(season_df) or active in _year_set(game_df)
    return SeasonCoverage(
        as_of_date=as_of_date,
        active_season=active,
        season_window=resolved_window,
        seasons_present=seasons_present,
        overlay_seasons=overlay_years,
        overlay_rows=int(len(overlay)),
        active_season_present=active_present,
        active_season_source=active_source,
        current_season_missing=not active_present,
        current_season_missing_reason=missing_reason if not active_present else None,
        sdio_in_season=sdio_in_season,
    )


def attach_team_coverage(player: SeasonCoverage, team: SeasonCoverage) -> SeasonCoverage:
    """Copy team-grain flags onto the published manifest. Does not fork the window.

    ``current_season_missing`` stays honest for FO desks: if active-season
    team rows are absent, the published flag is True even when player
    metrics already include Y.
    """
    player.team_seasons_present = list(team.seasons_present)
    player.team_overlay_seasons = list(team.overlay_seasons)
    player.team_overlay_rows = int(team.overlay_rows)
    player.team_active_season_present = bool(team.active_season_present)
    player.team_active_season_source = team.active_season_source
    player.team_current_season_missing = bool(team.current_season_missing)
    player.team_current_season_missing_reason = team.current_season_missing_reason
    if team.current_season_missing:
        player.current_season_missing = True
        if player.current_season_missing_reason is None:
            player.current_season_missing_reason = team.current_season_missing_reason
    return player


def select_sdio_team_overlay_frame(
    season_df: pd.DataFrame | None,
    game_df: pd.DataFrame | None,
    standings_df: pd.DataFrame | None = None,
    *,
    window: list[int],
    lahman_years: set[int],
) -> pd.DataFrame:
    """Reuse the player overlay selector; add game standings only for leftover years.

    One-day ``games_by_date`` standings are not merged onto a fat season-stat
    rollup (that would publish W=1 as a season record). Standings fill years
    that have no counting overlay at all.
    """
    overlay = select_sdio_overlay_frame(
        season_df, game_df, window=window, lahman_years=lahman_years
    )
    need = [year for year in window if year not in lahman_years]
    have = _year_set(overlay)
    still = [year for year in need if year not in have]
    if (
        still
        and standings_df is not None
        and not standings_df.empty
        and "year_id" in standings_df.columns
    ):
        years = pd.to_numeric(standings_df["year_id"], errors="coerce")
        extra = standings_df.loc[years.isin(still)]
        if not extra.empty:
            overlay = (
                pd.concat([overlay, extra], ignore_index=True)
                if not overlay.empty
                else extra.copy()
            )
    return overlay if overlay is not None else pd.DataFrame()


def _fill_team_identity(overlay: pd.DataFrame, lahman: pd.DataFrame) -> pd.DataFrame:
    """Prefer historical Lahman names / franchise ids when team_id already exists."""
    out = overlay.copy()
    if out.empty or lahman is None or lahman.empty or "team_id" not in out.columns:
        return out
    if "team_id" not in lahman.columns:
        return out
    sort_col = "year_id" if "year_id" in lahman.columns else lahman.columns[0]
    latest = lahman.sort_values(sort_col).groupby("team_id", as_index=False).last()
    rename = {}
    for src, dest in (
        ("team_name", "_lahman_team_name"),
        ("franchise_id", "_lahman_franchise_id"),
        ("league_id", "_lahman_league_id"),
    ):
        if src in latest.columns:
            rename[src] = dest
    latest = latest[["team_id", *rename.keys()]].rename(columns=rename)
    out = out.merge(latest, on="team_id", how="left")
    if "_lahman_team_name" in out.columns:
        out["team_name"] = out["_lahman_team_name"].combine_first(out.get("team_name"))
    if "_lahman_franchise_id" in out.columns:
        if "franchise_id" not in out.columns:
            out["franchise_id"] = pd.NA
        out["franchise_id"] = out["_lahman_franchise_id"].combine_first(out["franchise_id"])
    if "_lahman_league_id" in out.columns:
        if "league_id" not in out.columns:
            out["league_id"] = pd.NA
        out["league_id"] = out["_lahman_league_id"].combine_first(out["league_id"])
    return out.drop(columns=[c for c in out.columns if c.startswith("_lahman_")])


def _apply_team_counting_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """Approx team WAR from rolled-up counting stats. Never written to the spine."""
    out = frame.copy()
    if out.empty:
        return out
    if "team_total_war" not in out.columns:
        out["team_total_war"] = pd.NA
    missing = pd.to_numeric(out["team_total_war"], errors="coerce").isna()
    if missing.any():
        proxy = out.loc[missing].apply(approx_vs_replacement_from_counting, axis=1)
        out.loc[missing, "team_total_war"] = proxy
        if "team_batting_war" in out.columns:
            bat_missing = missing & pd.to_numeric(out["team_batting_war"], errors="coerce").isna()
            out.loc[bat_missing, "team_batting_war"] = out.loc[bat_missing, "team_total_war"]
        if "team_pitching_war" in out.columns:
            pit_missing = missing & pd.to_numeric(out["team_pitching_war"], errors="coerce").isna()
            out.loc[pit_missing, "team_pitching_war"] = 0.0
    if "war_source" not in out.columns:
        out["war_source"] = "approx"
    else:
        out["war_source"] = out["war_source"].fillna("approx")
    return out


def _derive_team_onfield(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill run_diff / Pythag from RS/RA/G when SDIO has them. No invented payroll."""
    out = frame.copy()
    if out.empty:
        return out
    rs = pd.to_numeric(out["runs_scored"], errors="coerce") if "runs_scored" in out.columns else pd.Series(pd.NA, index=out.index)
    ra = pd.to_numeric(out["runs_allowed"], errors="coerce") if "runs_allowed" in out.columns else pd.Series(pd.NA, index=out.index)
    games = pd.to_numeric(out["games"], errors="coerce") if "games" in out.columns else pd.Series(pd.NA, index=out.index)
    wins = pd.to_numeric(out["wins"], errors="coerce") if "wins" in out.columns else pd.Series(pd.NA, index=out.index)
    war = pd.to_numeric(out["team_total_war"], errors="coerce") if "team_total_war" in out.columns else pd.Series(pd.NA, index=out.index)

    if "run_diff" not in out.columns:
        out["run_diff"] = pd.NA
    run_missing = pd.to_numeric(out["run_diff"], errors="coerce").isna()
    out.loc[run_missing, "run_diff"] = (rs - ra)[run_missing]

    score_mask = rs.notna() & ra.notna() & games.notna() & (games > 0)
    if score_mask.any():
        pythag = pythagorean_wins(rs.fillna(0), ra.fillna(0), games.fillna(0))
        if "pythag_wins" not in out.columns:
            out["pythag_wins"] = pd.NA
        pythag_missing = pd.to_numeric(out["pythag_wins"], errors="coerce").isna()
        out.loc[score_mask & pythag_missing, "pythag_wins"] = pythag[score_mask & pythag_missing]
        if "pythag_gap" not in out.columns:
            out["pythag_gap"] = pd.NA
        gap_mask = score_mask & wins.notna() & pd.to_numeric(out["pythag_gap"], errors="coerce").isna()
        if gap_mask.any():
            out.loc[gap_mask, "pythag_gap"] = pythag_gap(wins, pythag)[gap_mask]

    if "war_win_gap" not in out.columns:
        out["war_win_gap"] = pd.NA
    war_mask = wins.notna() & war.notna() & pd.to_numeric(out["war_win_gap"], errors="coerce").isna()
    if war_mask.any():
        out.loc[war_mask, "war_win_gap"] = war_win_gap(wins, war)[war_mask]
    return out


def _project_team_publish_columns(overlay: pd.DataFrame, lahman: pd.DataFrame) -> pd.DataFrame:
    cols = list(
        dict.fromkeys(
            [
                *(lahman.columns.tolist() if lahman is not None and not lahman.empty else []),
                *TEAM_PUBLISH_COLUMNS,
                "stat_source",
            ]
        )
    )
    out = overlay.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = pd.NA
    return out[cols]


def bridge_sdio_player_season_metrics(
    lahman_df: pd.DataFrame,
    season_df: pd.DataFrame | None = None,
    game_df: pd.DataFrame | None = None,
    *,
    as_of_date: str,
    window: list[int] | None = None,
    extract_report: dict | None = None,
) -> tuple[pd.DataFrame, SeasonCoverage]:
    """UNION SDIO seasons onto Lahman metrics for years Lahman does not have.

    Historical Lahman + BR WAR rows stay intact. Overlay rows get a counting
    proxy marked ``war_source=approx`` so cards can rank the active season.
    That proxy is never written back onto ``player_season_stat``.
    """
    resolved_window = list(window) if window else default_season_window(as_of_date)
    lahman = lahman_df.copy() if lahman_df is not None else pd.DataFrame()
    lahman_years = _year_set(lahman)
    overlay = select_sdio_overlay_frame(
        season_df, game_df, window=resolved_window, lahman_years=lahman_years
    )
    if not overlay.empty:
        overlay = _apply_counting_proxy(overlay)
        if "stat_source" not in overlay.columns:
            overlay["stat_source"] = "sportsdataio"
        combined = pd.concat([lahman, overlay], ignore_index=True) if not lahman.empty else overlay
    else:
        overlay = pd.DataFrame()
        combined = lahman

    coverage = coverage_from_frames(
        as_of_date=as_of_date,
        window=resolved_window,
        combined=combined,
        overlay=overlay,
        lahman_years=lahman_years,
        source_present=_source_frames_present(season_df, game_df),
        extract_report=extract_report,
        season_df=season_df,
        game_df=game_df,
    )
    return combined, coverage


def bridge_sdio_team_season_metrics(
    lahman_df: pd.DataFrame,
    season_df: pd.DataFrame | None = None,
    game_df: pd.DataFrame | None = None,
    standings_df: pd.DataFrame | None = None,
    *,
    as_of_date: str,
    window: list[int] | None = None,
) -> tuple[pd.DataFrame, SeasonCoverage]:
    """UNION SDIO team-season rows onto Lahman team metrics for missing window years.

    Overlay vs new publish path: same ``team_onfield_contract_metrics.csv`` FO
    already loads. Historical Lahman + BR team rows stay intact. Overlay WAR is
    ``approx``; payroll is left null. Nothing is written back onto the SDIO spine.
    """
    resolved_window = list(window) if window else default_season_window(as_of_date)
    lahman = lahman_df.copy() if lahman_df is not None else pd.DataFrame()
    lahman_years = _year_set(lahman)
    overlay = select_sdio_team_overlay_frame(
        season_df,
        game_df,
        standings_df,
        window=resolved_window,
        lahman_years=lahman_years,
    )
    if not overlay.empty:
        overlay = _fill_team_identity(overlay, lahman)
        overlay = _apply_team_counting_proxy(overlay)
        overlay = _derive_team_onfield(overlay)
        if "stat_source" not in overlay.columns:
            overlay["stat_source"] = "sportsdataio"
        overlay = _project_team_publish_columns(overlay, lahman)
        combined = pd.concat([lahman, overlay], ignore_index=True) if not lahman.empty else overlay
    else:
        overlay = pd.DataFrame()
        combined = lahman

    coverage = coverage_from_frames(
        as_of_date=as_of_date,
        window=resolved_window,
        combined=combined,
        overlay=overlay,
        lahman_years=lahman_years,
        source_present=_source_frames_present(season_df, game_df, standings_df),
        season_df=season_df,
        game_df=game_df,
    )
    return combined, coverage


def write_metrics_manifest(artifacts_dir: Path, coverage: SeasonCoverage) -> Path:
    dest = Path(artifacts_dir) / METRICS_MANIFEST_NAME
    dest.write_text(json.dumps(coverage.to_dict(), indent=2) + "\n", encoding="utf-8")
    return dest


def load_sdio_metric_frames(con: duckdb.DuckDBPyConnection) -> SdioPublishFrames:
    frames = SdioPublishFrames()
    if _table_has_rows(con, "player_season_stat"):
        frames.player_season = con.execute(_SDIO_PLAYER_SEASON_QUERY).fetchdf()
        log.info("SportsDataIO player_season_stat available — %d season rows", len(frames.player_season))
        frames.team_season = con.execute(_SDIO_TEAM_SEASON_QUERY).fetchdf()
        log.info("SportsDataIO team season rollup — %d rows", len(frames.team_season))
    if _table_has_rows(con, "player_game_stat"):
        frames.player_game = con.execute(_SDIO_PLAYER_GAME_ROLLUP_QUERY).fetchdf()
        log.info("SportsDataIO player_game_stat available — %d rolled-up rows", len(frames.player_game))
        frames.team_game = con.execute(_SDIO_TEAM_GAME_ROLLUP_QUERY).fetchdf()
        log.info("SportsDataIO team game rollup — %d rows", len(frames.team_game))
    if _table_has_rows(con, "game"):
        frames.team_standings = con.execute(_SDIO_TEAM_STANDINGS_QUERY).fetchdf()
        log.info("SportsDataIO team standings from game — %d rows", len(frames.team_standings))
    return frames


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
    sdio_frames = load_sdio_metric_frames(con)

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

    as_of = default_as_of_date()
    window = seasons_from_settings(settings, as_of)
    extract_report = read_raw_payload(
        endpoint=ENDPOINT_EXTRACT_REPORT,
        as_of_date=as_of,
        filename="extract_report.json",
        raw_dir=settings["raw_dir"],
    )

    # ---- Team exports (Lahman + SDIO overlay for window years Lahman lacks) ----
    team_df, team_coverage = bridge_sdio_team_season_metrics(
        team_df,
        sdio_frames.team_season,
        sdio_frames.team_game,
        sdio_frames.team_standings,
        as_of_date=as_of,
        window=window,
    )
    if team_coverage.current_season_missing:
        log.warning(
            "Active season %s is missing from team_onfield_contract_metrics (%s). "
            "Prior-only team metrics are not current-year.",
            team_coverage.active_season,
            team_coverage.current_season_missing_reason or "unknown",
        )
    else:
        log.info(
            "Active team season %s present via %s; overlay seasons=%s",
            team_coverage.active_season,
            team_coverage.active_season_source,
            team_coverage.overlay_seasons,
        )

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
    player_df, coverage = bridge_sdio_player_season_metrics(
        player_df,
        sdio_frames.player_season,
        sdio_frames.player_game,
        as_of_date=as_of,
        window=window,
        extract_report=extract_report if isinstance(extract_report, dict) else None,
    )
    coverage = attach_team_coverage(coverage, team_coverage)
    if coverage.current_season_missing:
        log.warning(
            "Active season %s is missing from player_season_metrics (%s). "
            "Prior-only metrics are not current-year.",
            coverage.active_season,
            coverage.current_season_missing_reason or "unknown",
        )
    else:
        log.info(
            "Active season %s present via %s; overlay seasons=%s",
            coverage.active_season,
            coverage.active_season_source,
            coverage.overlay_seasons,
        )
    player_df = enrich_player_season_phase0(player_df, as_of_date=as_of)
    player_df.to_csv(artifacts_dir / "player_season_metrics.csv", index=False)
    log.info("Wrote player_season_metrics.csv (%d rows)", len(player_df))
    write_metrics_manifest(artifacts_dir, coverage)
    log.info(
        "Wrote %s (current_season_missing=%s team_current_season_missing=%s)",
        METRICS_MANIFEST_NAME,
        coverage.current_season_missing,
        coverage.team_current_season_missing,
    )
    cards_path = emit_ranked_fantasy_cards(
        artifacts_dir, as_of_date=as_of, player_df=player_df
    )
    log.info("Wrote %s from player_season_metrics", cards_path.relative_to(artifacts_dir))

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
