from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    disambiguating_player_id_columns,
    render_plotly_chart,
)


def test_compute_slider_max_handles_empty_years() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_uses_later_of_data_and_current_year() -> None:
    assert compute_slider_max([2016], 2026) == 2026
    assert compute_slider_max([2024, 2027], 2026) == 2027


def test_render_plotly_chart_delegates_to_streamlit_once() -> None:
    fig = go.Figure()
    streamlit = Mock()

    render_plotly_chart(
        fig,
        streamlit,
        {"paper_bgcolor": "#0d1117"},
        height=512,
    )

    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.height == 512


def test_disambiguating_player_id_columns_only_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["one", "two", "three"],
            "name_full": ["Alex Smith", "Alex Smith", "Jamie Jones"],
        }
    )

    assert disambiguating_player_id_columns(players) == ["player_id"]


def test_disambiguating_player_id_columns_omits_unique_or_unidentifiable_names() -> None:
    unique_names = pd.DataFrame(
        {
            "player_id": ["one", "two"],
            "name_full": ["Alex Smith", "Jamie Jones"],
        }
    )
    no_player_id = pd.DataFrame({"name_full": ["Alex Smith", "Alex Smith"]})

    assert disambiguating_player_id_columns(unique_names) == []
    assert disambiguating_player_id_columns(no_player_id) == []
