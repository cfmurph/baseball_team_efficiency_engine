from __future__ import annotations

from collections.abc import Iterable
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


def compute_slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a slider upper bound that is safe for empty metric extracts."""
    sorted_years = sorted(int(year) for year in years)
    if not sorted_years:
        return int(current_year)
    return max(sorted_years[-1], int(current_year))


def apply_plotly_layout(fig: Any, layout: dict[str, Any] = PLOTLY_LAYOUT) -> None:
    """Apply the dashboard's shared Plotly layout to a figure."""
    fig.update_layout(**layout)


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    *,
    height: int = 400,
    layout: dict[str, Any] = PLOTLY_LAYOUT,
) -> None:
    """Apply layout and render a Plotly figure through Streamlit."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
