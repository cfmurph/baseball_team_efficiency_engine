"""Small, testable helpers for the Streamlit dashboard."""
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
    xaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        tickfont=dict(color="#8b949e", size=11),
    ),
    yaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        tickfont=dict(color="#8b949e", size=11),
    ),
    legend=dict(
        bgcolor="#161b22",
        bordercolor="#21262d",
        borderwidth=1,
        font=dict(size=11, color="#c9d1d9"),
    ),
    margin=dict(t=40, b=30, l=10, r=10),
    colorway=["#bf1c20", "#1f6feb", "#3fb950", "#d29922", "#a371f7", "#f78166", "#58a6ff"],
)

SCATTER_MARKER = dict(size=7, opacity=0.75, line=dict(width=0.5, color="#0d1117"))


def apply_layout(fig: Any) -> None:
    """Apply the Baseball Savant dark layout to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(
    fig: Any,
    plotly_chart: Callable[..., None],
    height: int = 400,
) -> None:
    """Apply dashboard chart styling, then render via the supplied Streamlit hook."""
    apply_layout(fig)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)


def max_selectable_year(
    years: Iterable[int],
    current_year: int | None = None,
) -> int:
    """Return a safe dashboard slider upper bound for available metric years."""
    year_values = list(years)
    if current_year is None:
        current_year = datetime.date.today().year
    return max(max(year_values), current_year) if year_values else current_year


def has_name_collision(players: pd.DataFrame) -> bool:
    """Return True when the current view contains repeated display names."""
    return "name_full" in players.columns and bool(players.duplicated("name_full", keep=False).any())


def player_id_columns_for_name_collisions(players: pd.DataFrame) -> list[str]:
    """Show player IDs only when they disambiguate same-name players."""
    if has_name_collision(players) and "player_id" in players.columns:
        return ["player_id"]
    return []
