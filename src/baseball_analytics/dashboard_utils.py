from __future__ import annotations

import datetime
from collections.abc import Iterable
from typing import Any

import pandas as pd


def slider_max(years: Iterable[int | float], current_year: int | None = None) -> int:
    """Return a safe dashboard slider upper bound."""
    if current_year is None:
        current_year = datetime.date.today().year

    year_values = pd.Series(list(years)).dropna()
    if year_values.empty:
        return int(current_year)

    return max(int(year_values.astype(int).max()), int(current_year))


def scale_payroll_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll and salary display columns from dollars to millions."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "salary" in df.columns:
        df["salary"] = df["salary"] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Return the player id column when visible names are ambiguous."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []


def render_plotly_chart(streamlit_module: Any, fig: Any, layout: dict[str, Any], height: int = 400) -> None:
    """Apply dashboard chart defaults and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
