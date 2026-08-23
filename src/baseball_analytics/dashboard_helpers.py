from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def compute_slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for year sliders."""
    if not all_years:
        return int(current_year)
    return max(int(all_years[-1]), int(current_year))


def apply_layout_and_render_chart(
    fig: Any,
    *,
    apply_layout: Callable[[Any], None],
    plotly_chart: Callable[..., Any],
    height: int = 400,
) -> None:
    """Apply layout styling and render a Plotly figure."""
    apply_layout(fig)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)
