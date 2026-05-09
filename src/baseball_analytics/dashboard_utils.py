from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


def compute_slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a slider upper bound that remains valid when no metric years exist."""
    year_list = [int(year) for year in years]
    return max(max(year_list), int(current_year)) if year_list else int(current_year)


def render_plotly_chart(fig: Any, streamlit_module: Any, layout: dict, height: int = 400) -> None:
    """Apply dashboard chart defaults and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def player_id_columns_for_name_collisions(df: pd.DataFrame) -> list[str]:
    """Show player_id when same-name players would otherwise be ambiguous."""
    if "name_full" not in df.columns or "player_id" not in df.columns:
        return []
    return ["player_id"] if df.duplicated("name_full", keep=False).any() else []
