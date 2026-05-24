from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def compute_slider_max(years: Sequence[int], current_year: int) -> int:
    """Return a safe upper bound for year sliders."""
    if not years:
        return int(current_year)
    return max(max(int(year) for year in years), int(current_year))


def apply_plotly_layout(fig: Any, layout: dict[str, Any]) -> None:
    """Apply the dashboard's shared Plotly layout to a figure."""
    fig.update_layout(**layout)


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    layout: dict[str, Any],
    *,
    height: int = 400,
) -> None:
    """Apply dashboard styling and render a Plotly figure through Streamlit."""
    apply_plotly_layout(fig, layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def disambiguating_player_id_columns(
    players: pd.DataFrame,
    *,
    name_column: str = "name_full",
    id_column: str = "player_id",
) -> list[str]:
    """Return the player ID column when same-name rows need disambiguation."""
    if name_column not in players.columns or id_column not in players.columns:
        return []
    if players.duplicated(name_column, keep=False).any():
        return [id_column]
    return []
