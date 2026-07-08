from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def slider_max_year(all_years: Sequence[int], current_year: int) -> int:
    """Return a slider upper bound that is safe for empty data and future years."""
    return max(int(all_years[-1]), int(current_year)) if all_years else int(current_year)


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw dollars to display-scale values."""
    scaled = df.copy()
    for col in [
        "payroll",
        "max_salary",
        "median_salary",
        "payroll_per_win",
        "cost_per_war",
        "surplus_value",
    ]:
        if col in scaled.columns:
            scaled[col] = scaled[col] / 1_000_000
    if "salary" in scaled.columns:
        scaled["salary"] = scaled["salary"] / 1_000_000
    if "dead_money_share" in scaled.columns:
        scaled["dead_money_share"] = scaled["dead_money_share"] * 100
    return scaled


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name players would otherwise be ambiguous."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []


def apply_plotly_layout(fig: Any, layout: dict[str, Any]) -> None:
    """Apply the dashboard's standard Plotly layout in one testable place."""
    fig.update_layout(**layout)


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    layout: dict[str, Any],
    height: int = 400,
) -> None:
    """Apply layout and delegate rendering to Streamlit's Plotly chart API."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
