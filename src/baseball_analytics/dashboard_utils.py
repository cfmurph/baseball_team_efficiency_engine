from __future__ import annotations

from collections.abc import Sequence
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
    legend=dict(bgcolor="#161b22", bordercolor="#21262d", borderwidth=1, font=dict(size=11, color="#c9d1d9")),
    margin=dict(t=40, b=30, l=10, r=10),
    colorway=["#bf1c20", "#1f6feb", "#3fb950", "#d29922", "#a371f7", "#f78166", "#58a6ff"],
)

SCATTER_MARKER = dict(size=7, opacity=0.75, line=dict(width=0.5, color="#0d1117"))


def slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a dashboard slider max that remains valid when no years exist."""
    return max(max(all_years), current_year) if all_years else current_year


def apply_plotly_layout(fig: Any) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(fig: Any, streamlit_module: Any, height: int = 400) -> None:
    """Apply dashboard Plotly styling and render through Streamlit exactly once."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> tuple[list[str], bool]:
    """Return player-id display columns when same-name players appear in a view."""
    has_name_collision = name_col in df.columns and df.duplicated(name_col, keep=False).any()
    id_cols = [id_col] if has_name_collision and id_col in df.columns else []
    return id_cols, bool(has_name_collision)
