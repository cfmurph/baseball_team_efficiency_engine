from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def slider_max(all_years: Iterable[int], current_year: int) -> int:
    """Return a safe max year for dashboard sliders."""
    years = list(all_years)
    if not years:
        return int(current_year)
    return max(int(years[-1]), int(current_year))


def player_id_columns_for_name_collisions(df: Any) -> list[str]:
    """Show player_id when the filtered view contains same-name players."""
    if "name_full" not in df.columns or "player_id" not in df.columns:
        return []
    return ["player_id"] if df.duplicated("name_full", keep=False).any() else []


def render_plotly_chart(fig: Any, streamlit_module: Any, layout: dict[str, Any], height: int = 400) -> None:
    """Apply the dashboard Plotly layout and render through Streamlit."""
    fig.update_layout(**layout)
    fig.update_layout(height=height)
    streamlit_module.plotly_chart(fig, use_container_width=True)
