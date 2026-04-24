from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def compute_slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for season sliders."""
    return max(all_years[-1], current_year) if all_years else current_year


def render_chart(
    fig: Any,
    apply_layout: Callable[[Any], None],
    plotly_chart: Callable[..., None],
    height: int = 400,
) -> None:
    """Apply layout and render a figure in Streamlit."""
    apply_layout(fig)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)
