from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd


def compute_slider_max(years: Iterable[int], today: dt.date | None = None) -> int:
    """Return the upper year bound for dashboard range sliders."""
    current_year = (today or dt.date.today()).year
    year_values = sorted(int(year) for year in years)
    return max(year_values[-1], current_year) if year_values else current_year


def render_plotly_chart(streamlit_module, fig, height: int, layout: dict) -> None:
    """Apply dashboard Plotly layout and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)


def player_id_columns_for_name_collisions(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Show player IDs only when same-name players appear in the current view."""
    if name_col not in df.columns or id_col not in df.columns:
        return []
    return [id_col] if df.duplicated(name_col, keep=False).any() else []
