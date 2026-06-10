from __future__ import annotations

import datetime as _dt
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd


def slider_max(years: Iterable[int], current_year: int | None = None) -> int:
    """Return the largest selectable dashboard year.

    The dashboard should remain usable before artifacts contain any seasons and
    should allow current/future years when generated data lags the calendar.
    """
    current = current_year if current_year is not None else _dt.date.today().year
    year_list = sorted(int(year) for year in years)
    return max(year_list[-1], current) if year_list else current


def has_name_collisions(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Return True when the current table view contains repeated player names."""
    return name_col in df.columns and df.duplicated(name_col, keep=False).any()


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name players need disambiguation."""
    if has_name_collisions(df, name_col=name_col) and id_col in df.columns:
        return [id_col]
    return []


def render_plotly_chart(
    fig: Any,
    renderer: Callable[..., Any],
    *,
    height: int = 400,
    layout: dict[str, Any] | None = None,
) -> None:
    """Apply shared Plotly layout settings, then call the provided renderer."""
    if layout is not None:
        fig.update_layout(**layout)
    fig.update_layout(height=height)
    renderer(fig, use_container_width=True)
