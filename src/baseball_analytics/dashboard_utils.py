from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Callable

import pandas as pd


def slider_max(years: Iterable[int], current_year: int) -> int:
    """Return a safe max year for Streamlit sliders, even with no data."""
    year_list = list(years)
    return max(year_list[-1], current_year) if year_list else current_year


def has_name_collision(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Detect whether a filtered table contains multiple rows with the same name."""
    return name_col in df.columns and bool(df.duplicated(name_col, keep=False).any())


def collision_id_columns(
    df: pd.DataFrame,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Return the ID column prefix needed to disambiguate same-name players."""
    return [id_col] if has_name_collision(df, name_col) and id_col in df.columns else []


def render_plotly_chart(
    fig: Any,
    plotly_chart: Callable[..., Any],
    layout: dict[str, Any],
    height: int = 400,
) -> Any:
    """Apply dashboard layout and render the figure through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    return plotly_chart(fig, use_container_width=True)
