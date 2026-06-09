from __future__ import annotations

from collections.abc import Callable, Sequence
import datetime as dt

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


def slider_max(years: Sequence[int], current_year: int | None = None) -> int:
    """Return a safe Streamlit slider upper bound for metric years."""
    if current_year is None:
        current_year = dt.date.today().year
    return max(max(years), current_year) if years else current_year


def apply_plotly_dark_layout(fig) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(fig, renderer: Callable[..., None], height: int = 400) -> None:
    """Apply dashboard layout and render a Plotly chart through Streamlit."""
    apply_plotly_dark_layout(fig)
    fig.update_layout(height=height)
    renderer(fig, use_container_width=True)


def has_name_collision(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Return True when the visible player table contains shared display names."""
    return name_col in df.columns and df.duplicated(name_col, keep=False).any()


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Prepend player_id only when same-name players need disambiguation."""
    return [id_col] if has_name_collision(df, name_col) and id_col in df.columns else []
