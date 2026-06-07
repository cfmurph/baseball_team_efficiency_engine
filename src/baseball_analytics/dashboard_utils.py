"""Small dashboard helpers that are safe to unit test without Streamlit startup."""
from __future__ import annotations

import datetime
from collections.abc import Iterable

import pandas as pd


MONEY_DISPLAY_COLUMNS = (
    "payroll",
    "max_salary",
    "median_salary",
    "payroll_per_win",
    "cost_per_war",
    "surplus_value",
    "salary",
)

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e6edf3", size=12),
    title_font=dict(size=14, color="#e6edf3", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#30363d", tickfont=dict(color="#8b949e", size=11)),
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1, font=dict(size=11, color="#c9d1d9")),
    margin=dict(t=40, b=30, l=10, r=10),
    colorway=["#bf1c20", "#1f6feb", "#3fb950", "#d29922", "#a371f7", "#f78166", "#58a6ff"],
)

SCATTER_MARKER = dict(size=7, opacity=0.75, line=dict(width=0.5, color="#0d1117"))


def scale_payroll_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw dollar/share columns to display units without mutating input."""
    display = df.copy()
    for col in MONEY_DISPLAY_COLUMNS:
        if col in display.columns:
            display[col] = display[col] / 1_000_000
    if "dead_money_share" in display.columns:
        display["dead_money_share"] = display["dead_money_share"] * 100
    return display


def slider_max_year(years: Iterable[object], current_year: int | None = None) -> int:
    """Return a slider maximum that handles empty data and future artifact years."""
    current = current_year if current_year is not None else datetime.date.today().year
    valid_years = [int(year) for year in years if pd.notna(year)]
    return max(max(valid_years), current) if valid_years else current


def apply_plotly_layout(fig) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(streamlit_module, fig, height: int = 400) -> None:
    """Apply layout/height, then render through Streamlit's Plotly API."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def has_duplicate_names(df: pd.DataFrame, name_column: str = "name_full") -> bool:
    """Return True when a filtered player view contains same-name players."""
    return name_column in df.columns and df.duplicated(name_column, keep=False).any()


def player_id_columns_for_collisions(
    df: pd.DataFrame,
    name_column: str = "name_full",
    id_column: str = "player_id",
) -> list[str]:
    """Show player IDs only when they disambiguate same-name players."""
    if has_duplicate_names(df, name_column) and id_column in df.columns:
        return [id_column]
    return []
