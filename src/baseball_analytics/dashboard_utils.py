from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def compute_slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for dashboard year sliders."""
    if not all_years:
        return current_year
    return max(max(all_years), current_year)


def render_plotly_chart(
    streamlit_api: Any,
    fig: Any,
    *,
    height: int = 400,
    apply_layout: Callable[[Any], None] | None = None,
) -> None:
    """Apply dashboard chart defaults and delegate rendering to Streamlit."""
    if apply_layout is not None:
        apply_layout(fig)
    fig.update_layout(height=height)
    streamlit_api.plotly_chart(fig, use_container_width=True)
