"""Small dashboard helpers that are safe to unit test without Streamlit startup."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a slider upper bound even when metrics artifacts contain no years."""
    if not all_years:
        return int(current_year)
    return max(int(max(all_years)), int(current_year))


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    layout: Mapping[str, Any],
    height: int = 400,
) -> None:
    """Apply the shared layout and delegate rendering to Streamlit's Plotly API."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
