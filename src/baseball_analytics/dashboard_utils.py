from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def compute_slider_max(years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for dashboard year sliders."""
    clean_years = [int(year) for year in years if pd.notna(year)]
    if not clean_years:
        return int(current_year)
    return max(max(clean_years), int(current_year))


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    layout: dict[str, Any],
    height: int = 400,
) -> None:
    """Apply shared layout and render a Plotly figure through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def has_player_name_collision(
    df: pd.DataFrame,
    name_col: str = "name_full",
) -> bool:
    """Return True when the current player view contains repeated names."""
    return name_col in df.columns and bool(df.duplicated(name_col, keep=False).any())


def player_id_prefix_for_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Include player_id first only when same-name players need disambiguation."""
    if id_col not in df.columns:
        return []
    if not has_player_name_collision(df, name_col=name_col):
        return []
    return [id_col]


def columns_with_player_id_for_collisions(
    df: pd.DataFrame,
    columns: Sequence[str],
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Prefix display columns with player_id when same-name rows are present."""
    prefix = player_id_prefix_for_collisions(df, name_col=name_col, id_col=id_col)
    return prefix + [column for column in columns if column not in prefix]
