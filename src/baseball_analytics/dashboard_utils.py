from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd


def slider_max_for_years(years: Sequence[int], current_year: int) -> int:
    """Return a safe Streamlit slider maximum for available seasons."""
    if not years:
        return current_year
    return max(int(years[-1]), int(current_year))


def apply_plotly_layout(fig: Any, layout: dict[str, Any]) -> None:
    """Apply a shared Plotly layout to a figure."""
    fig.update_layout(**layout)


def render_plotly_chart(
    fig: Any,
    renderer: Callable[..., Any],
    *,
    layout: dict[str, Any],
    height: int = 400,
) -> None:
    """Apply dashboard layout and render a Plotly chart exactly once."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    renderer(fig, use_container_width=True)


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player_id when the current view contains distinct players sharing a name."""
    if "name_full" not in df.columns or "player_id" not in df.columns:
        return []

    collision_counts = (
        df[["name_full", "player_id"]]
        .dropna(subset=["name_full", "player_id"])
        .drop_duplicates()
        .groupby("name_full")["player_id"]
        .nunique()
    )
    return ["player_id"] if (collision_counts > 1).any() else []
