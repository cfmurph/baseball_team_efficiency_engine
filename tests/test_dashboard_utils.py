from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    frontier_line_points,
    has_duplicate_player_names,
    player_id_columns_for_name_collision,
    render_plotly_chart,
)


def test_compute_slider_max_handles_empty_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_includes_future_artifact_year() -> None:
    assert compute_slider_max([2024, 2027], current_year=2026) == 2027


def test_player_id_columns_added_only_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "name_full": ["Chris Young", "Chris Young", "Aaron Judge"],
            "player_id": ["youngch01", "youngch02", "judgeaa01"],
        }
    )

    assert has_duplicate_player_names(players)
    assert player_id_columns_for_name_collision(players) == ["player_id"]


def test_player_id_columns_not_added_for_unique_names() -> None:
    players = pd.DataFrame(
        {
            "name_full": ["Aaron Judge", "Gerrit Cole"],
            "player_id": ["judgeaa01", "colege01"],
        }
    )

    assert not has_duplicate_player_names(players)
    assert player_id_columns_for_name_collision(players) == []


def test_player_id_columns_handle_missing_player_id() -> None:
    players = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert has_duplicate_player_names(players)
    assert player_id_columns_for_name_collision(players) == []


def test_frontier_line_points_are_sorted_and_deduplicated() -> None:
    frontier_data = pd.DataFrame(
        {
            "payroll_m": [120.0, 90.0, 90.0, 150.0],
            "frontier_pred": [88.0, 81.0, 81.0, 95.0],
            "wins": [87, 80, 80, 94],
        }
    )

    points = frontier_line_points(frontier_data)

    assert points["payroll_m"].tolist() == [90.0, 120.0, 150.0]
    assert points["frontier_pred"].tolist() == [81.0, 88.0, 95.0]


def test_render_plotly_chart_applies_layout_height_and_renders_once() -> None:
    fig = go.Figure()
    streamlit = Mock()

    render_plotly_chart(
        fig,
        streamlit,
        layout={"paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117"},
        height=512,
    )

    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    assert fig.layout.height == 512
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
