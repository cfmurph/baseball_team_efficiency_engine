from __future__ import annotations

import numpy as np
import pandas as pd


def pythagorean_wins(runs_scored: pd.Series, runs_allowed: pd.Series, games: pd.Series, exponent: float = 1.83) -> pd.Series:
    numerator = np.power(runs_scored, exponent)
    denominator = numerator + np.power(runs_allowed, exponent)
    pct = np.where(denominator == 0, np.nan, numerator / denominator)
    return pct * games


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where((denominator == 0) | pd.isna(denominator), np.nan, numerator / denominator)


def salary_concentration(df: pd.DataFrame, salary_col: str = "salary") -> pd.Series:
    values = df[salary_col].dropna().sort_values().to_numpy()
    if len(values) == 0:
        return pd.Series({"gini_salary": np.nan})
    index = np.arange(1, len(values) + 1)
    gini = ((2 * np.sum(index * values)) / (len(values) * np.sum(values))) - ((len(values) + 1) / len(values))
    return pd.Series({"gini_salary": gini})


def top_salary_shares(group: pd.DataFrame, salary_col: str = "salary") -> pd.Series:
    salaries = group[salary_col].dropna().sort_values(ascending=False)
    total = salaries.sum()
    if total == 0:
        return pd.Series({"top_1_salary_share": np.nan, "top_3_salary_share": np.nan, "top_5_salary_share": np.nan})
    return pd.Series({
        "top_1_salary_share": salaries.head(1).sum() / total,
        "top_3_salary_share": salaries.head(3).sum() / total,
        "top_5_salary_share": salaries.head(5).sum() / total,
    })
