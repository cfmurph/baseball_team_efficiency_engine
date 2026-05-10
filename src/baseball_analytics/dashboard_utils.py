from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def compute_slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a safe max year for dashboard sliders."""
    if not all_years:
        return int(current_year)
    return max(int(all_years[-1]), int(current_year))


def render_plotly_chart(
    streamlit_api: Any,
    fig: Any,
    layout: dict[str, Any],
    height: int = 400,
) -> None:
    """Apply dashboard layout and render a Plotly chart through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_api.plotly_chart(fig, use_container_width=True)
