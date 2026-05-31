from __future__ import annotations

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


def slider_max_year(all_years: Sequence[int], current_year: int) -> int:
    """Return the maximum season value for year-range sliders."""
    if not all_years:
        return current_year
    return max(int(all_years[-1]), int(current_year))


def apply_plotly_layout(fig) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(
    fig,
    renderer: Callable[..., object],
    *,
    height: int = 400,
) -> None:
    """Style and render a Plotly figure through the provided Streamlit renderer."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    renderer(fig, use_container_width=True)


def has_same_name_players(df: pd.DataFrame) -> bool:
    return "name_full" in df.columns and bool(df.duplicated("name_full", keep=False).any())


def player_id_columns_for_collisions(df: pd.DataFrame) -> list[str]:
    if has_same_name_players(df) and "player_id" in df.columns:
        return ["player_id"]
    return []
