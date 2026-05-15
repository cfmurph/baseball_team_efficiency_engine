from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


def compute_slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a safe max value for dashboard year sliders."""
    year_values = [int(year) for year in years]
    if not year_values:
        return int(current_year)
    return max(max(year_values), int(current_year))


def has_duplicate_player_names(players: pd.DataFrame) -> bool:
    """Whether the current player view contains distinct rows sharing a name."""
    return "name_full" in players.columns and players.duplicated("name_full", keep=False).any()


def player_id_columns_for_name_collision(players: pd.DataFrame) -> list[str]:
    """Show player_id only when duplicate names need disambiguation."""
    if has_duplicate_player_names(players) and "player_id" in players.columns:
        return ["player_id"]
    return []


def frontier_line_points(frontier_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare sorted unique points for the efficiency frontier line trace."""
    return (
        frontier_data.sort_values("payroll_m")[["payroll_m", "frontier_pred"]]
        .drop_duplicates()
    )


def render_plotly_chart(
    fig: Any,
    streamlit_module: Any,
    *,
    layout: Mapping[str, Any],
    height: int = 400,
) -> None:
    """Apply dashboard Plotly styling and render a chart once."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
