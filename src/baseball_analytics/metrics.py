from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def pythagorean_wins(
    runs_scored: pd.Series,
    runs_allowed: pd.Series,
    games: pd.Series,
    exponent: float = 1.83,
) -> pd.Series:
    numerator = np.power(runs_scored, exponent)
    denominator = numerator + np.power(runs_allowed, exponent)
    pct = np.where(denominator == 0, np.nan, numerator / denominator)
    return pct * games


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where((denominator == 0) | pd.isna(denominator), np.nan, numerator / denominator)


# ---------------------------------------------------------------------------
# Salary concentration
# ---------------------------------------------------------------------------

def salary_concentration(df: pd.DataFrame, salary_col: str = "salary") -> pd.Series:
    values = df[salary_col].dropna().sort_values().to_numpy()
    if len(values) == 0:
        return pd.Series({"gini_salary": np.nan})
    index = np.arange(1, len(values) + 1)
    gini = (
        (2 * np.sum(index * values)) / (len(values) * np.sum(values))
    ) - ((len(values) + 1) / len(values))
    return pd.Series({"gini_salary": gini})


def top_salary_shares(group: pd.DataFrame, salary_col: str = "salary") -> pd.Series:
    salaries = group[salary_col].dropna().sort_values(ascending=False)
    total = salaries.sum()
    if total == 0:
        return pd.Series(
            {"top_1_salary_share": np.nan, "top_3_salary_share": np.nan, "top_5_salary_share": np.nan}
        )
    return pd.Series(
        {
            "top_1_salary_share": salaries.head(1).sum() / total,
            "top_3_salary_share": salaries.head(3).sum() / total,
            "top_5_salary_share": salaries.head(5).sum() / total,
        }
    )


# ---------------------------------------------------------------------------
# WAR-based efficiency metrics  (team-season level)
# ---------------------------------------------------------------------------

_MARKET_RATE_WAR = 8_000_000.0   # $ per WAR on open market (rough modern baseline)


def cost_per_war(payroll: pd.Series, total_war: pd.Series) -> pd.Series:
    """Dollars spent per WAR produced."""
    return safe_divide(payroll, total_war)


def war_per_dollar(total_war: pd.Series, payroll: pd.Series, scale: float = 1_000_000.0) -> pd.Series:
    """WAR produced per $1 M of payroll."""
    return safe_divide(total_war * scale, payroll)


def surplus_value_team(payroll: pd.Series, total_war: pd.Series, market_rate: float = _MARKET_RATE_WAR) -> pd.Series:
    """
    Team surplus value = (WAR * market_rate_per_war) - payroll.
    Positive → team extracted value above open-market cost.
    """
    return total_war * market_rate - payroll


# ---------------------------------------------------------------------------
# Over/Under-performance
# ---------------------------------------------------------------------------

def pythag_gap(wins: pd.Series, pythag_wins: pd.Series) -> pd.Series:
    """Actual wins minus Pythagorean expected wins."""
    return wins - pythag_wins


def war_win_gap(wins: pd.Series, total_war: pd.Series, replacement_wins: float = 48.0) -> pd.Series:
    """
    Wins above replacement implied by WAR vs actual wins.
    replacement_wins: A 0-WAR roster is historically ~48 wins (162 * 0.294).
    """
    war_implied = total_war + replacement_wins
    return wins - war_implied


# ---------------------------------------------------------------------------
# Roster construction
# ---------------------------------------------------------------------------

def payroll_underperformer_share(
    player_df: pd.DataFrame,
    salary_col: str = "salary",
    war_col: str = "player_war",
    war_threshold: float = 0.5,
) -> pd.Series:
    """
    Fraction of payroll tied to players with WAR < threshold (dead money proxy).
    Expects a player-season DataFrame for a single team-season.
    """
    underperf = player_df[player_df[war_col] < war_threshold][salary_col].sum()
    total = player_df[salary_col].sum()
    if total == 0:
        return np.nan
    return underperf / total


def war_concentration(
    player_df: pd.DataFrame,
    war_col: str = "player_war",
    top_n: int = 3,
) -> float:
    """Fraction of team WAR provided by top-N players."""
    vals = player_df[war_col].dropna().sort_values(ascending=False)
    total = vals.sum()
    if total <= 0:
        return np.nan
    return vals.head(top_n).sum() / total


# ---------------------------------------------------------------------------
# Contract efficiency (player level)
# ---------------------------------------------------------------------------

def player_surplus_value(
    salary: pd.Series,
    player_war: pd.Series,
    market_rate: float = _MARKET_RATE_WAR,
) -> pd.Series:
    """
    Surplus value per player-season = (WAR * market_rate) - salary.
    Positive → team is getting a bargain.
    Negative → team is overpaying (dead money if WAR ≤ 0).
    """
    return player_war * market_rate - salary


def classify_contract(surplus: pd.Series, war: pd.Series) -> pd.Series:
    """
    Label contracts based on WAR and surplus value:
    - 'dead_money'      : WAR ≤ 0 and salary > 0
    - 'overpaid'        : WAR > 0 but surplus < 0
    - 'fair_value'      : surplus ≈ 0 (within ±$2M)
    - 'surplus_value'   : surplus > $2M
    - 'pre_arb_value'   : low salary, positive WAR (not modeled separately here)
    """
    conditions = [
        (war <= 0),
        (surplus < -2_000_000),
        (surplus.abs() <= 2_000_000),
        (surplus > 2_000_000),
    ]
    choices = ["dead_money", "overpaid", "fair_value", "surplus_value"]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=surplus.index)


# ---------------------------------------------------------------------------
# Window detection (team trajectory)
# ---------------------------------------------------------------------------

def detect_team_window(
    team_history: pd.DataFrame,
    win_col: str = "wins",
    payroll_col: str = "payroll",
    war_col: str = "team_total_war",
    window: int = 3,
) -> pd.DataFrame:
    """
    Classify each team-season as: 'contending', 'declining', 'rebuilding', or 'developing'.

    Rules (applied per team, rolling window):
      - contending  : wins ≥ 88 AND payroll trend is flat or rising
      - declining   : wins trending down > 5 over window AND payroll high
      - rebuilding  : wins < 75 AND payroll in bottom third
      - developing  : wins trending up > 5 over window
      - other       : else → 'steady'
    """
    df = team_history.sort_values("year_id").copy()

    df["wins_rolling"] = df[win_col].rolling(window, min_periods=1).mean()
    df["wins_delta"] = df[win_col].diff(window).fillna(0)
    df["payroll_pct"] = df[payroll_col].rank(pct=True)

    conditions = [
        (df[win_col] >= 88),
        (df["wins_delta"] < -5) & (df["payroll_pct"] > 0.5),
        (df[win_col] < 75) & (df["payroll_pct"] < 0.33),
        (df["wins_delta"] > 5),
    ]
    choices = ["contending", "declining", "rebuilding", "developing"]
    df["window_phase"] = np.select(conditions, choices, default="steady")
    return df
