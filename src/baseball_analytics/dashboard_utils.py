from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dollar and ratio columns to dashboard display units."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "salary" in df.columns:
        df["salary"] = df["salary"] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def compute_slider_max(years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for dashboard year sliders."""
    if not years:
        return current_year
    return max(max(int(year) for year in years), current_year)


def season_range_defaults(
    years: Sequence[int],
    current_year: int,
    trailing_years: int = 9,
) -> tuple[int, int, tuple[int, int]]:
    """Return min/max/default values for the season-compare range slider."""
    slider_max = compute_slider_max(years, current_year)
    if not years:
        return current_year, current_year, (current_year, current_year)

    min_year = min(int(year) for year in years)
    return min_year, slider_max, (max(min_year, slider_max - trailing_years), slider_max)


def id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name players appear in the current view."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []


def render_plotly_chart(fig, streamlit_module, layout: Mapping, height: int = 400) -> None:
    """Apply shared dashboard chart styling and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
