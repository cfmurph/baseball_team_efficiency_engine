from __future__ import annotations

import datetime
from collections.abc import Callable, Sequence

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


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw dollars to display units."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "salary" in df.columns:
        df["salary"] = df["salary"] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def slider_max(all_years: Sequence[int], current_year: int | None = None) -> int:
    """Return a safe upper bound for dashboard year sliders."""
    if current_year is None:
        current_year = datetime.date.today().year
    if not all_years:
        return current_year
    return max(int(all_years[-1]), current_year)


def apply_plotly_layout(fig) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_chart(fig, plotly_chart: Callable, height: int = 400) -> None:
    """Apply dashboard layout and render a Plotly chart with Streamlit."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player_id when duplicate display names refer to distinct player rows."""
    has_name_collision = (
        "name_full" in df.columns
        and df["name_full"].duplicated(keep=False).any()
    )
    if has_name_collision and "player_id" in df.columns:
        return ["player_id"]
    return []
