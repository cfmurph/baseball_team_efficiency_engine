from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def compute_slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a safe max value for year sliders, even when no data is loaded."""
    year_list = list(years)
    if not year_list:
        return current_year
    return max(max(year_list), current_year)


def apply_plotly_layout(fig: Any, layout: dict[str, Any]) -> None:
    """Apply the shared dashboard layout to a Plotly-like figure."""
    fig.update_layout(**layout)


def render_plotly_chart(fig: Any, streamlit_module: Any, layout: dict[str, Any], height: int = 400) -> None:
    """Apply dashboard chart styling, then delegate rendering to Streamlit."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def has_name_collisions(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Return True when a visible player table contains duplicate display names."""
    return name_col in df.columns and bool(df.duplicated(name_col, keep=False).any())


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Expose player IDs only when duplicate names would otherwise be ambiguous."""
    if has_name_collisions(df, name_col=name_col) and id_col in df.columns:
        return [id_col]
    return []
