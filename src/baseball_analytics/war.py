"""
Lahman-derived WAR approximations.

Batting WAR uses a wOBA → wRAA → batting runs chain calibrated to Lahman
column availability.  Pitching WAR uses FIP-based runs allowed.  These are
intentional approximations — they are not FanGraphs/Baseball-Reference WAR —
but are well-correlated and sufficient for team-level efficiency analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants (calibrated to modern era; adjust via settings if needed)
# ---------------------------------------------------------------------------
WOBA_WEIGHTS = {
    "wBB": 0.69,
    "wHBP": 0.72,
    "w1B": 0.88,
    "w2B": 1.24,
    "w3B": 1.56,
    "wHR": 2.06,
}
LEAGUE_WOBA = 0.320
WOBA_SCALE = 1.20          # league avg wOBA / league avg OBP ≈ 1.15-1.25
RUNS_PER_WIN = 9.5         # historical average; 10 in modern era
REPLACEMENT_LEVEL_BWAR = 2.0   # replacement-level wins above avg per 600 PA
REPLACEMENT_LEVEL_PWAR = 2.0   # per 200 IP


# ---------------------------------------------------------------------------
# Batting WAR
# ---------------------------------------------------------------------------

def batting_war(batting: pd.DataFrame) -> pd.DataFrame:
    """
    Compute approximate batting WAR per player-season.

    Parameters
    ----------
    batting : DataFrame with Lahman Batting columns present.

    Returns
    -------
    DataFrame with columns [playerID, yearID, teamID, batting_war]
    """
    b = batting.copy()

    # Plate appearances (Lahman doesn't have PA directly; AB + BB + HBP + SF + SH)
    for col in ["BB", "HBP", "SF", "SH", "IBB"]:
        if col not in b.columns:
            b[col] = 0
    b[["BB", "HBP", "SF", "SH", "IBB"]] = b[["BB", "HBP", "SF", "SH", "IBB"]].fillna(0)
    b["PA"] = b["AB"] + b["BB"] + b["HBP"].fillna(0) + b["SF"].fillna(0) + b["SH"].fillna(0)

    # Hit components — Lahman CSV via Rdatasets uses X2B/X3B
    b["H"] = b["H"].fillna(0)
    # Support both naming conventions
    if "2B" not in b.columns and "X2B" in b.columns:
        b["2B"] = b["X2B"]
    if "3B" not in b.columns and "X3B" in b.columns:
        b["3B"] = b["X3B"]
    b["2B"] = b["2B"].fillna(0)
    b["3B"] = b["3B"].fillna(0)
    b["HR"] = b["HR"].fillna(0)
    b["1B"] = b["H"] - b["2B"] - b["3B"] - b["HR"]

    # wOBA numerator
    b["wOBA_num"] = (
        WOBA_WEIGHTS["wBB"] * (b["BB"] - b["IBB"])
        + WOBA_WEIGHTS["wHBP"] * b["HBP"]
        + WOBA_WEIGHTS["w1B"] * b["1B"]
        + WOBA_WEIGHTS["w2B"] * b["2B"]
        + WOBA_WEIGHTS["w3B"] * b["3B"]
        + WOBA_WEIGHTS["wHR"] * b["HR"]
    )
    # Denominator: AB + BB - IBB + SF + HBP
    b["wOBA_den"] = b["AB"] + (b["BB"] - b["IBB"]) + b["SF"] + b["HBP"]
    b["wOBA"] = np.where(b["wOBA_den"] > 0, b["wOBA_num"] / b["wOBA_den"], np.nan)

    # wRAA: runs above average
    b["wRAA"] = np.where(
        b["PA"] > 0,
        ((b["wOBA"] - LEAGUE_WOBA) / WOBA_SCALE) * b["PA"],
        0.0,
    )

    # Positional / replacement adjustment (flat; requires fielding for full version)
    # Grant every batter replacement-level credit proportional to PA
    b["rep_runs"] = REPLACEMENT_LEVEL_BWAR * RUNS_PER_WIN * (b["PA"] / 600.0)

    # WAR = (wRAA + rep_runs) / runs_per_win
    b["batting_war"] = (b["wRAA"] + b["rep_runs"]) / RUNS_PER_WIN

    out = (
        b.groupby(["playerID", "yearID", "teamID"], as_index=False)
        .agg(
            batting_war=("batting_war", "sum"),
            pa=("PA", "sum"),
            woba=("wOBA", "mean"),
            hr=("HR", "sum"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Pitching WAR (FIP-based)
# ---------------------------------------------------------------------------

LEAGUE_FIP_CONSTANT = 3.20     # FIP constant calibrated to approximate ERA


def pitching_war(pitching: pd.DataFrame) -> pd.DataFrame:
    """
    Compute approximate pitching WAR per player-season using FIP.

    Parameters
    ----------
    pitching : DataFrame with Lahman Pitching columns present.

    Returns
    -------
    DataFrame with columns [playerID, yearID, teamID, pitching_war]
    """
    p = pitching.copy()

    for col in ["BB", "HBP", "HR", "SO", "IPouts"]:
        if col not in p.columns:
            p[col] = 0
    p[["BB", "HBP", "HR", "SO", "IPouts"]] = p[["BB", "HBP", "HR", "SO", "IPouts"]].fillna(0)

    # IP from IPouts (IPouts = outs recorded)
    p["IP"] = p["IPouts"] / 3.0

    # FIP = (13*HR + 3*(BB+HBP) - 2*SO) / IP + FIP_constant
    p["FIP"] = np.where(
        p["IP"] > 0,
        (13 * p["HR"] + 3 * (p["BB"] + p["HBP"]) - 2 * p["SO"]) / p["IP"] + LEAGUE_FIP_CONSTANT,
        np.nan,
    )

    # FIP-based RA/9 vs league avg RA/9 → runs prevented
    league_ra9 = 4.50   # approximate modern league average
    p["fip_runs_prevented"] = np.where(
        p["IP"] > 0,
        (league_ra9 - p["FIP"]) * (p["IP"] / 9.0),
        0.0,
    )

    # Replacement level: grant credit proportional to IP
    p["rep_runs"] = REPLACEMENT_LEVEL_PWAR * RUNS_PER_WIN * (p["IP"] / 200.0)

    p["pitching_war"] = (p["fip_runs_prevented"] + p["rep_runs"]) / RUNS_PER_WIN

    out = (
        p.groupby(["playerID", "yearID", "teamID"], as_index=False)
        .agg(
            pitching_war=("pitching_war", "sum"),
            ip=("IP", "sum"),
            fip=("FIP", "mean"),
            era=("ERA", "mean") if "ERA" in p.columns else ("pitching_war", "count"),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Team-level WAR aggregation
# ---------------------------------------------------------------------------

def team_war_totals(batting_war_df: pd.DataFrame, pitching_war_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll batting + pitching WAR up to team-season level.

    Returns DataFrame with [yearID, teamID, team_batting_war, team_pitching_war, team_total_war]
    """
    bat = (
        batting_war_df
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(team_batting_war=("batting_war", "sum"))
    )
    pit = (
        pitching_war_df
        .groupby(["yearID", "teamID"], as_index=False)
        .agg(team_pitching_war=("pitching_war", "sum"))
    )
    merged = bat.merge(pit, on=["yearID", "teamID"], how="outer").fillna(0)
    merged["team_total_war"] = merged["team_batting_war"] + merged["team_pitching_war"]
    return merged


# ---------------------------------------------------------------------------
# BaseRuns (Smyth model) — team level
# ---------------------------------------------------------------------------

def base_runs(
    hits: pd.Series,
    singles: pd.Series,
    doubles: pd.Series,
    triples: pd.Series,
    hr: pd.Series,
    bb: pd.Series,
    hbp: pd.Series,
    ab: pd.Series,
    sf: pd.Series,
) -> pd.Series:
    """
    Approximate BaseRuns using the Smyth/Baumer-Albert model.

    Expected runs = A * B / (B + C) + HR
    where:
        A = baserunners
        B = advancement factor
        C = outs
        HR = home runs
    """
    A = hits + bb + hbp - hr
    B = (
        1.4 * singles
        + 2.34 * doubles
        + 3.01 * triples
        + 1.89 * hr
        + 0.44 * (bb + hbp)
        - 0.07 * (bb + hbp)  # intentional walk adjustment (simplified)
    )
    C = ab - hits + sf
    denom = B + C
    base_r = np.where(denom > 0, A * B / denom + hr, np.nan)
    return pd.Series(base_r, index=hits.index)
