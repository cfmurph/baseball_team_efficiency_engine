"""Tests for dashboard helper behavior that guards recent UI regressions."""
from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    has_name_collision,
    max_selectable_year,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
)


def test_render_plotly_chart_applies_layout_and_calls_streamlit_renderer():
    fig = go.Figure()
    plotly_chart = Mock()

    render_plotly_chart(fig, plotly_chart, height=512)

    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_max_selectable_year_uses_current_year_when_no_data_years():
    assert max_selectable_year([], current_year=2026) == 2026


def test_max_selectable_year_preserves_future_data_year():
    assert max_selectable_year([2016, 2028], current_year=2026) == 2028


def test_player_id_columns_are_shown_only_for_same_name_players():
    players = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "jonesad01"],
            "name_full": ["John Smith", "John Smith", "Adam Jones"],
        }
    )

    assert has_name_collision(players)
    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_hidden_without_name_collision():
    players = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert not has_name_collision(players)
    assert player_id_columns_for_name_collisions(players) == []
