from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def compute_slider_max(all_years: Sequence[int], current_year: int) -> int:
    """Pick the highest year for slider upper bound with empty-input guard."""
    return max(all_years[-1], current_year) if all_years else current_year


def render_chart(
    fig: Any,
    *,
    apply_layout: Callable[[Any], None],
    render: Callable[..., Any],
    height: int = 400,
) -> None:
    """Apply shared layout and render exactly once."""
    apply_layout(fig)
    fig.update_layout(height=height)
    render(fig, use_container_width=True)
