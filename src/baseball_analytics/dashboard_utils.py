from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd


def compute_slider_max(all_years: Iterable[int], current_year: int) -> int:
    """Return the upper bound for dashboard year sliders."""
    years = [int(year) for year in all_years]
    if not years:
        return int(current_year)
    return max(max(years), int(current_year))


def has_name_collision(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Return True when a display table contains multiple rows with the same name."""
    return name_col in df.columns and bool(df.duplicated(name_col, keep=False).any())


def player_id_prefix_columns(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Return player_id as a leading column only when names need disambiguation."""
    if id_col in df.columns and has_name_collision(df, name_col=name_col):
        return [id_col]
    return []


def render_plotly_chart(
    fig: Any,
    plotly_chart: Callable[..., Any],
    layout: Mapping[str, Any],
    height: int = 400,
) -> None:
    """Apply shared layout and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    plotly_chart(fig, use_container_width=True)
