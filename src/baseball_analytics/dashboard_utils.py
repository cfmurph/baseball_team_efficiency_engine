from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


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


def slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a Streamlit-safe max year even when no metrics rows exist."""
    year_list = list(years)
    if not year_list:
        return current_year
    return max(max(year_list), current_year)


def apply_plotly_layout(fig: Any) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)


def render_plotly_chart(
    fig: Any,
    *,
    plotly_chart: Callable[..., Any],
    height: int = 400,
) -> None:
    """Apply dashboard chart styling and delegate rendering to Streamlit."""
    apply_plotly_layout(fig)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)
