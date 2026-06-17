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


def calculate_slider_max(years: Iterable[int], current_year: int | None = None) -> int:
    """Return the max year a dashboard slider should expose."""
    current = current_year if current_year is not None else datetime.date.today().year
    year_values = [int(year) for year in years]
    return max(max(year_values), current) if year_values else current


def player_id_columns(df: pd.DataFrame, name_col: str = "name_full", id_col: str = "player_id") -> list[str]:
    """Show player IDs only when same-name players need disambiguation."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []


def render_plotly_chart(fig: Any, plotly_chart: Callable[..., Any], height: int = 400) -> None:
    """Apply dashboard chart styling and delegate rendering to Streamlit."""
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)
