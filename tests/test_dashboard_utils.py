from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    has_name_collisions,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max,
)


def test_slider_max_uses_current_year_when_no_metric_years() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2024, 2025, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_height_and_calls_renderer_once() -> None:
    fig = go.Figure()
    renderer = Mock()

    render_plotly_chart(
        fig,
        renderer,
        height=460,
        layout={"template": "plotly_dark", "paper_bgcolor": "#0d1117"},
    )

    assert fig.layout.height == 460
    assert fig.layout.paper_bgcolor == "#0d1117"
    renderer.assert_called_once_with(fig, use_container_width=True)


def test_player_id_column_is_added_only_for_same_name_players() -> None:
    duplicate_names = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02"],
            "name_full": ["John Smith", "John Smith"],
        }
    )
    unique_names = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert has_name_collisions(duplicate_names)
    assert player_id_columns_for_name_collisions(duplicate_names) == ["player_id"]
    assert not has_name_collisions(unique_names)
    assert player_id_columns_for_name_collisions(unique_names) == []


def test_player_id_column_requires_player_id_field() -> None:
    duplicate_names = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert has_name_collisions(duplicate_names)
    assert player_id_columns_for_name_collisions(duplicate_names) == []
