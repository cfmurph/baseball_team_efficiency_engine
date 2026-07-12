from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def calculate_slider_max(
    years: Iterable[int],
    current_year: int | None = None,
) -> int:
    """Return the upper slider bound, including future/current calendar years."""
    resolved_current_year = current_year or datetime.date.today().year
    year_list = sorted(int(year) for year in years)
    if not year_list:
        return resolved_current_year
    return max(year_list[-1], resolved_current_year)


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw dollars to display units."""
    scaled = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in scaled.columns:
            scaled[col] = scaled[col] / 1_000_000
    if "salary" in scaled.columns:
        scaled["salary"] = scaled["salary"] / 1_000_000
    if "dead_money_share" in scaled.columns:
        scaled["dead_money_share"] = scaled["dead_money_share"] * 100
    return scaled


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player_id only when same-name players need disambiguation."""
    if "name_full" not in df.columns or "player_id" not in df.columns:
        return []
    if df.duplicated("name_full", keep=False).any():
        return ["player_id"]
    return []


def has_player_name_collision(df: pd.DataFrame) -> bool:
    if "name_full" not in df.columns:
        return False
    return bool(df.duplicated("name_full", keep=False).any())


def apply_plotly_layout(fig: Any, layout: Mapping[str, Any]) -> None:
    fig.update_layout(**layout)


def render_plotly_chart(fig: Any, streamlit_module: Any, layout: Mapping[str, Any], height: int = 400) -> None:
    """Apply the shared layout and delegate rendering to Streamlit."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
