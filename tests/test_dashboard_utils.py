from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    calculate_slider_max,
    player_id_columns,
    render_plotly_chart,
)


def test_calculate_slider_max_uses_current_year_for_empty_metrics() -> None:
    assert calculate_slider_max([], current_year=2026) == 2026


def test_calculate_slider_max_keeps_future_data_selectable() -> None:
    assert calculate_slider_max([2020, 2028], current_year=2026) == 2028


def test_render_plotly_chart_styles_and_delegates_once() -> None:
    fig = go.Figure()
    plotly_chart = MagicMock()

    render_plotly_chart(fig, plotly_chart, height=512)

    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_only_when_same_name_players_need_disambiguation() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns(players) == ["player_id"]
    assert player_id_columns(players.drop_duplicates("name_full")) == []
    assert player_id_columns(players.drop(columns=["player_id"])) == []
