from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def slider_max_from_years(
    years: Iterable[int],
    current_year: int | None = None,
) -> int:
    """Return the upper bound for dashboard year sliders."""
    if current_year is None:
        current_year = datetime.date.today().year

    parsed_years = [int(year) for year in years]
    if not parsed_years:
        return current_year
    return max(max(parsed_years), current_year)


def has_name_collisions(
    df: pd.DataFrame,
    name_column: str = "name_full",
) -> bool:
    """Return True when a filtered player table contains duplicate names."""
    return name_column in df.columns and bool(df.duplicated(name_column, keep=False).any())


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_column: str = "name_full",
    id_column: str = "player_id",
) -> list[str]:
    """Show player IDs only when duplicate names would otherwise be ambiguous."""
    if has_name_collisions(df, name_column) and id_column in df.columns:
        return [id_column]
    return []


def render_plotly_chart(
    fig: Any,
    renderer: Any,
    layout: Mapping[str, Any],
    height: int = 400,
) -> None:
    """Apply dashboard layout and render a Plotly chart via the Streamlit API."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    renderer.plotly_chart(fig, use_container_width=True)
