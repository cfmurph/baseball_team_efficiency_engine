from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd


def compute_slider_max(years: Iterable[Any], current_year: int) -> int:
    """Return the upper slider bound, including the current calendar year."""
    clean_years = sorted(int(year) for year in years if pd.notna(year))
    if not clean_years:
        return int(current_year)
    return max(clean_years[-1], int(current_year))


def render_plotly_chart(fig: Any, streamlit_api: Any, layout: dict[str, Any], height: int = 400) -> None:
    """Apply dashboard layout and delegate rendering to Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_api.plotly_chart(fig, use_container_width=True)


def has_name_collision(df: pd.DataFrame, name_col: str = "name_full") -> bool:
    """Return True when the visible rows contain multiple players with a name."""
    return name_col in df.columns and bool(df.duplicated(name_col, keep=False).any())


def player_id_disambiguation_columns(
    df: pd.DataFrame,
    base_columns: Sequence[str] | None = None,
    name_col: str = "name_full",
    id_col: str = "player_id",
) -> list[str]:
    """Prepend player_id when needed to distinguish same-name players."""
    columns = list(base_columns or [])
    if has_name_collision(df, name_col=name_col) and id_col in df.columns:
        return [id_col, *columns]
    return columns
