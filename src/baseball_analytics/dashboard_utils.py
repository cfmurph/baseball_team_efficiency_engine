from __future__ import annotations

import datetime
from collections.abc import Sequence

import pandas as pd


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw dollars to millions for display."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "salary" in df.columns:
        df["salary"] = df["salary"] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def year_slider_max(all_years: Sequence[int], today: datetime.date | None = None) -> int:
    """Return the upper year bound, falling back to the current year for empty metrics."""
    current_year = (today or datetime.date.today()).year
    if not all_years:
        return current_year
    return max(max(int(year) for year in all_years), current_year)


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player_id only when same-name players need disambiguation."""
    if "name_full" not in df.columns or "player_id" not in df.columns:
        return []
    return ["player_id"] if df.duplicated("name_full", keep=False).any() else []


def render_plotly_chart(fig, st_module, layout: dict, height: int = 400) -> None:
    """Apply dashboard layout and render a Plotly figure through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    st_module.plotly_chart(fig, use_container_width=True)
