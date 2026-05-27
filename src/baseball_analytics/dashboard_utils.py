from __future__ import annotations

import pandas as pd


def scale_payroll_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dollar columns to display units without mutating input data."""
    scaled = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in scaled.columns:
            scaled[col] = scaled[col] / 1_000_000
    if "salary" in scaled.columns:
        scaled["salary"] = scaled["salary"] / 1_000_000
    if "dead_money_share" in scaled.columns:
        scaled["dead_money_share"] = scaled["dead_money_share"] * 100
    return scaled


def slider_max_year(years: list[int], current_year: int) -> int:
    """Use the later of the latest data year and current year, or current year if data is empty."""
    return max(years[-1], current_year) if years else current_year


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name players would otherwise be ambiguous."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []


def apply_plotly_layout(fig, layout: dict) -> None:
    fig.update_layout(**layout)


def render_plotly_chart(fig, streamlit_module, layout: dict, height: int = 400) -> None:
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
