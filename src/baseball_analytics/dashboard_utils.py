from __future__ import annotations

import datetime
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


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
    """Convert payroll/salary columns from raw dollars to dashboard display units."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "salary" in df.columns:
        df["salary"] = df["salary"] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def slider_max_year(years: Iterable[Any], current_year: int | None = None) -> int:
    """Return the largest selectable season without failing on empty metrics."""
    if current_year is None:
        current_year = datetime.date.today().year

    year_values = sorted(int(year) for year in years if pd.notna(year))
    return max(year_values[-1], current_year) if year_values else current_year


def apply_plotly_layout(fig: Any) -> None:
    """Apply the dashboard's Baseball Savant dark layout to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(
    fig: Any,
    render_func: Callable[..., Any],
    height: int = 400,
) -> None:
    """Apply dashboard layout and render a Plotly figure exactly once."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    render_func(fig, use_container_width=True)
