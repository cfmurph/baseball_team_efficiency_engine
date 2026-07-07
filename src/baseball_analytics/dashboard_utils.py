from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def slider_max_year(all_years: Iterable[int], current_year: int) -> int:
    """Return a safe upper bound for year sliders when metric data is empty."""
    years = list(all_years)
    if not years:
        return int(current_year)
    return max(int(years[-1]), int(current_year))


def has_duplicate_player_names(df: pd.DataFrame) -> bool:
    """Detect same-name players in the current filtered view."""
    return "name_full" in df.columns and bool(df.duplicated("name_full", keep=False).any())


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player IDs only when needed to disambiguate same-name players."""
    if has_duplicate_player_names(df) and "player_id" in df.columns:
        return ["player_id"]
    return []


def render_plotly_chart(fig: Any, st_module: Any, layout: dict[str, Any], height: int = 400) -> None:
    """Apply the dashboard Plotly layout and render through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    st_module.plotly_chart(fig, use_container_width=True)
