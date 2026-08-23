"""
WAR: Baseball-Reference rWAR overlay + Lahman approximations as fallback.

Real WAR
--------
Baseball-Reference publishes daily batting/pitching WAR files at player ×
year × team grain (war_daily_bat.txt / war_daily_pitch.txt).  Those IDs are
bbref IDs, which Lahman exposes as People.bbrefID — that is the primary
player-ID crosswalk.  Team abbreviations differ (NYY vs NYA); see
data/crosswalks/br_team_map.csv.

Approximate WAR (fallback)
--------------------------
Batting WAR uses a wOBA → wRAA → batting runs chain calibrated to Lahman
column availability.  Pitching WAR uses FIP-based runs allowed.  Used when
a player-season-team row has no matching rWAR row.  ``war_source`` is
``real`` or ``approx`` so downstream metrics can tell which was used.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

WAR_SOURCE_REAL = "real"
WAR_SOURCE_APPROX = "approx"
WAR_SOURCE_MIXED = "mixed"

BR_BAT_FILENAME = "war_daily_bat.txt"
BR_PIT_FILENAME = "war_daily_pitch.txt"
DEFAULT_TEAM_MAP = Path("data/crosswalks/br_team_map.csv")


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


# ---------------------------------------------------------------------------
# Real WAR (Baseball-Reference rWAR)
# ---------------------------------------------------------------------------

def load_br_team_map(path: str | Path | None = None) -> pd.DataFrame:
    """Load the BR → Lahman team-ID crosswalk (year-aware)."""
    map_path = Path(path) if path else DEFAULT_TEAM_MAP
    if not map_path.exists():
        log.warning("BR team map not found at %s — team IDs will be used as-is", map_path)
        return pd.DataFrame(columns=["br_team_id", "lahman_team_id", "year_start", "year_end"])
    return pd.read_csv(map_path)


def map_br_team_ids(
    df: pd.DataFrame,
    team_map: pd.DataFrame,
    team_col: str = "team_ID",
    year_col: str = "year_ID",
) -> pd.Series:
    """
    Map Baseball-Reference team_ID to Lahman teamID using a year-aware crosswalk.

    Unmapped teams fall back to the BR abbreviation unchanged.
    """
    if team_map is None or team_map.empty:
        return df[team_col].astype(str)

    mapped = df[[team_col, year_col]].copy()
    mapped["_row"] = np.arange(len(mapped))
    merged = mapped.merge(
        team_map,
        left_on=team_col,
        right_on="br_team_id",
        how="left",
    )
    in_range = merged["year_start"].isna() | (
        (merged[year_col] >= merged["year_start"]) & (merged[year_col] <= merged["year_end"])
    )
    merged = merged.loc[in_range].drop_duplicates("_row", keep="first")
    merged = mapped[["_row"]].merge(merged, on="_row", how="left")
    result = merged["lahman_team_id"].where(merged["lahman_team_id"].notna(), merged[team_col])
    return result.astype(str)


def map_br_player_ids(war: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """
    Attach Lahman playerID to a BR WAR frame.

    Primary: People.bbrefID == BR player_ID (handles the ~500 IDs that differ).
    Fallback: BR player_ID used as playerID (they match for the large majority).
    """
    out = war.copy()
    if "player_ID" not in out.columns:
        raise ValueError("BR WAR frame must include player_ID")

    if people is not None and "bbrefID" in people.columns and "playerID" in people.columns:
        xw = (
            people.loc[
                people["bbrefID"].notna() & (people["bbrefID"].astype(str).str.len() > 0),
                ["playerID", "bbrefID"],
            ]
            .drop_duplicates("bbrefID")
        )
        out = out.merge(xw, left_on="player_ID", right_on="bbrefID", how="left")
        missing = out["playerID"].isna()
        out.loc[missing, "playerID"] = out.loc[missing, "player_ID"]
        out = out.drop(columns=["bbrefID"], errors="ignore")
    else:
        out["playerID"] = out["player_ID"]
    return out


def _read_br_war_file(path: Path) -> pd.DataFrame:
    """Read a BR war_daily_*.txt file; empty frame if missing/unreadable."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 — keep pipeline running on a bad file
        log.warning("Failed to read %s: %s", path, exc)
        return pd.DataFrame()
    if "WAR" in df.columns:
        df["WAR"] = pd.to_numeric(df["WAR"], errors="coerce")
    if "year_ID" in df.columns:
        df["year_ID"] = pd.to_numeric(df["year_ID"], errors="coerce")
    return df


def load_real_war(
    raw_dir: str | Path,
    people: pd.DataFrame,
    min_year: int,
    team_map_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load Baseball-Reference batting + pitching WAR and normalize to Lahman keys.

    Returns a DataFrame at player × year × team grain with columns:
        playerID, yearID, teamID, batting_war_real, pitching_war_real
    Empty DataFrame if the extract files are not present.
    """
    raw_dir = Path(raw_dir)
    bat = _read_br_war_file(raw_dir / BR_BAT_FILENAME)
    pit = _read_br_war_file(raw_dir / BR_PIT_FILENAME)
    if bat.empty and pit.empty:
        log.warning(
            "No BR WAR files in %s (run python3 -m pipeline.extract.pull_war)",
            raw_dir,
        )
        return pd.DataFrame(
            columns=["playerID", "yearID", "teamID", "batting_war_real", "pitching_war_real"]
        )

    team_map = load_br_team_map(team_map_path)

    frames = []
    if not bat.empty:
        bat = bat[bat["year_ID"] >= min_year].copy()
        bat = map_br_player_ids(bat, people)
        bat["teamID"] = map_br_team_ids(bat, team_map)
        bat = (
            bat.groupby(["playerID", "year_ID", "teamID"], as_index=False)
            .agg(batting_war_real=("WAR", "sum"))
        )
        frames.append(bat.rename(columns={"year_ID": "yearID"}))

    if not pit.empty:
        pit = pit[pit["year_ID"] >= min_year].copy()
        pit = map_br_player_ids(pit, people)
        pit["teamID"] = map_br_team_ids(pit, team_map)
        pit = (
            pit.groupby(["playerID", "year_ID", "teamID"], as_index=False)
            .agg(pitching_war_real=("WAR", "sum"))
        )
        frames.append(pit.rename(columns={"year_ID": "yearID"}))

    real = frames[0]
    for extra in frames[1:]:
        real = real.merge(extra, on=["playerID", "yearID", "teamID"], how="outer")

    if "batting_war_real" not in real.columns:
        real["batting_war_real"] = np.nan
    if "pitching_war_real" not in real.columns:
        real["pitching_war_real"] = np.nan

    log.info(
        "Loaded real WAR: %d player-team-seasons (%d with batting, %d with pitching)",
        len(real),
        int(real["batting_war_real"].notna().sum()),
        int(real["pitching_war_real"].notna().sum()),
    )
    return real[["playerID", "yearID", "teamID", "batting_war_real", "pitching_war_real"]]


def _fallback_player_year(
    player_df: pd.DataFrame,
    leftover: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach leftover real-WAR rows that missed the team-ID join, but only when
    both sides have a unique player-year (avoids mis-allocating traded players).
    """
    if leftover.empty or player_df.empty:
        return player_df

    still = player_df["war_source"].eq(WAR_SOURCE_APPROX)
    left_n = leftover.groupby(["player_id", "season_key"]).size()
    left_u = leftover.set_index(["player_id", "season_key"])
    left_u = left_u.loc[left_n[left_n == 1].index].reset_index()

    dest_n = player_df.loc[still].groupby(["player_id", "season_key"]).size()
    dest_keys = dest_n[dest_n == 1].index

    if left_u.empty or len(dest_keys) == 0:
        return player_df

    attach = left_u.set_index(["player_id", "season_key"])
    attach = attach.loc[attach.index.isin(dest_keys)].reset_index()
    if attach.empty:
        return player_df

    out = player_df.copy()
    key_cols = ["player_id", "season_key"]
    out = out.merge(
        attach[key_cols + ["batting_war_real", "pitching_war_real"]],
        on=key_cols,
        how="left",
        suffixes=("", "_fb"),
    )
    for col in ["batting_war_real", "pitching_war_real"]:
        fb = f"{col}_fb"
        if fb in out.columns:
            out[col] = out[col].combine_first(out[fb])
            out = out.drop(columns=[fb])
    return out


def apply_real_war(
    player_df: pd.DataFrame,
    real_war: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Overlay real WAR onto approximate player-season rows.

    ``batting_war`` / ``pitching_war`` / ``player_war`` become the effective
    values (real when present, else the Lahman approximation).
    ``war_source`` is ``real`` if any real component was used, else ``approx``.
    """
    df = player_df.copy()
    if "batting_war" not in df.columns:
        df["batting_war"] = 0.0
    if "pitching_war" not in df.columns:
        df["pitching_war"] = 0.0

    df["war_source"] = WAR_SOURCE_APPROX

    if real_war is None or real_war.empty:
        df["player_war"] = df["batting_war"].fillna(0) + df["pitching_war"].fillna(0)
        return df

    real = real_war.rename(
        columns={
            "playerID": "player_id",
            "yearID": "season_key",
            "teamID": "team_id",
        }
    )
    keep = ["player_id", "season_key", "team_id", "batting_war_real", "pitching_war_real"]
    keep = [c for c in keep if c in real.columns]
    real = real[keep].copy()

    df = df.merge(real, on=["player_id", "season_key", "team_id"], how="left")
    if "batting_war_real" not in df.columns:
        df["batting_war_real"] = np.nan
    if "pitching_war_real" not in df.columns:
        df["pitching_war_real"] = np.nan

    # Team-ID miss fallback: unique player-year leftovers only
    used = df.loc[
        df["batting_war_real"].notna() | df["pitching_war_real"].notna(),
        ["player_id", "season_key", "team_id"],
    ]
    leftover = real.merge(used, on=["player_id", "season_key", "team_id"], how="left", indicator=True)
    leftover = leftover.loc[leftover["_merge"] == "left_only"].drop(columns="_merge")
    if not leftover.empty:
        df["war_source"] = np.where(
            df["batting_war_real"].notna() | df["pitching_war_real"].notna(),
            WAR_SOURCE_REAL,
            WAR_SOURCE_APPROX,
        )
        df = _fallback_player_year(df, leftover)

    use_bat = df["batting_war_real"].notna()
    use_pit = df["pitching_war_real"].notna()
    df.loc[use_bat, "batting_war"] = df.loc[use_bat, "batting_war_real"]
    df.loc[use_pit, "pitching_war"] = df.loc[use_pit, "pitching_war_real"]
    df["player_war"] = df["batting_war"].fillna(0) + df["pitching_war"].fillna(0)
    df["war_source"] = np.where(use_bat | use_pit, WAR_SOURCE_REAL, WAR_SOURCE_APPROX)
    return df


def team_war_from_players(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll effective player WAR up to team-season and label the source mix.

    Returns [yearID, teamID, team_batting_war, team_pitching_war, team_total_war, war_source]
    """
    p = player_df.copy()
    if "yearID" not in p.columns:
        if "season_key" in p.columns:
            p = p.rename(columns={"season_key": "yearID"})
        elif "year_id" in p.columns:
            p = p.rename(columns={"year_id": "yearID"})
    if "teamID" not in p.columns and "team_id" in p.columns:
        p = p.rename(columns={"team_id": "teamID"})

    grouped = (
        p.groupby(["yearID", "teamID"], as_index=False)
        .agg(
            team_batting_war=("batting_war", "sum"),
            team_pitching_war=("pitching_war", "sum"),
            team_total_war=("player_war", "sum"),
        )
    )

    if "war_source" in p.columns:
        src = (
            p.groupby(["yearID", "teamID"])["war_source"]
            .agg(lambda s: (
                WAR_SOURCE_REAL if set(s.dropna()) == {WAR_SOURCE_REAL}
                else WAR_SOURCE_APPROX if set(s.dropna()) == {WAR_SOURCE_APPROX}
                else WAR_SOURCE_MIXED
            ))
            .reset_index()
        )
        grouped = grouped.merge(src, on=["yearID", "teamID"], how="left")
    else:
        grouped["war_source"] = WAR_SOURCE_APPROX
    return grouped
