from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


_PAYROLL_DISPLAY_COLUMNS = (
    "payroll",
    "max_salary",
    "median_salary",
    "payroll_per_win",
    "cost_per_war",
    "surplus_value",
)
_SALARY_DISPLAY_COLUMNS = ("salary",)


def calculate_slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a safe max value for year sliders."""
    year_list = [int(year) for year in years]
    if not year_list:
        return int(current_year)
    return max(max(year_list), int(current_year))


def scale_payroll_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dollar and share columns to dashboard display units."""
    display = df.copy()
    for col in _PAYROLL_DISPLAY_COLUMNS:
        if col in display.columns:
            display[col] = display[col] / 1_000_000
    for col in _SALARY_DISPLAY_COLUMNS:
        if col in display.columns:
            display[col] = display[col] / 1_000_000
    if "dead_money_share" in display.columns:
        display["dead_money_share"] = display["dead_money_share"] * 100
    return display


def player_id_columns_for_duplicate_names(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name rows would otherwise be ambiguous."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    if not df.duplicated(name_col, keep=False).any():
        return []
    return [id_col]


def apply_plotly_layout(fig: Any, layout: Mapping[str, Any]) -> None:
    """Apply shared Plotly layout settings in-place."""
    fig.update_layout(**layout)


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    layout: Mapping[str, Any],
    height: int = 400,
) -> None:
    """Apply layout/height, then delegate rendering to Streamlit exactly once."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
