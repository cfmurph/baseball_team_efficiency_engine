from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max,
)


def test_slider_max_uses_current_year_for_empty_data() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2021, 2028], current_year=2026) == 2028


def test_player_id_columns_added_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smitha01", "smithb01", "jonesc01"],
            "name_full": ["Alex Smith", "Alex Smith", "Casey Jones"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_omitted_without_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smitha01", "jonesc01"],
            "name_full": ["Alex Smith", "Casey Jones"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == []


def test_render_plotly_chart_applies_layout_and_calls_streamlit_once() -> None:
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    streamlit = MagicMock()

    render_plotly_chart(fig, streamlit, height=321)

    assert fig.layout.height == 321
    assert fig.layout.paper_bgcolor == "#0d1117"
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
