from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd


_MONEY_COLUMNS = (
    "payroll",
    "max_salary",
    "median_salary",
    "payroll_per_win",
    "cost_per_war",
    "surplus_value",
    "salary",
)


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dollar columns to millions and ratio columns to display units."""
    display = df.copy()
    for col in _MONEY_COLUMNS:
        if col in display.columns:
            display[col] = display[col] / 1_000_000
    if "dead_money_share" in display.columns:
        display["dead_money_share"] = display["dead_money_share"] * 100
    return display


def calculate_slider_max(
    all_years: Sequence[int],
    current_year: int | None = None,
) -> int:
    """Return a safe max year for Streamlit sliders, even with empty artifacts."""
    fallback_year = current_year if current_year is not None else _dt.date.today().year
    if not all_years:
        return fallback_year
    return max(max(int(year) for year in all_years), fallback_year)


def id_column_for_name_collisions(
    df: pd.DataFrame,
    *,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show a player ID only when repeated display names need disambiguation."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    if df.duplicated(name_col, keep=False).any():
        return [id_col]
    return []


def render_plotly_chart(
    fig: Any,
    plotly_chart: Callable[..., Any],
    layout: Mapping[str, Any],
    *,
    height: int = 400,
) -> Any:
    """Apply shared layout and delegate rendering to Streamlit's chart function."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    return plotly_chart(fig, use_container_width=True)
