from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max,
)


def test_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2024, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_and_renders_once() -> None:
    fig = go.Figure()
    streamlit = Mock()

    render_plotly_chart(fig, streamlit, height=360)

    assert fig.layout.height == 360
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_are_added_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["griffke01", "griffke02", "judgeaa01"],
            "name_full": ["Ken Griffey", "Ken Griffey", "Aaron Judge"],
        }
    )

    id_cols, has_collision = player_id_columns_for_name_collisions(players)

    assert has_collision is True
    assert id_cols == ["player_id"]


def test_player_id_columns_not_added_when_duplicate_names_are_filtered_out() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["griffke01", "judgeaa01"],
            "name_full": ["Ken Griffey", "Aaron Judge"],
        }
    )

    id_cols, has_collision = player_id_columns_for_name_collisions(players)

    assert has_collision is False
    assert id_cols == []


def test_name_collision_caption_can_trigger_without_player_id_column() -> None:
    players = pd.DataFrame({"name_full": ["Alex Gonzalez", "Alex Gonzalez"]})

    id_cols, has_collision = player_id_columns_for_name_collisions(players)

    assert has_collision is True
    assert id_cols == []
