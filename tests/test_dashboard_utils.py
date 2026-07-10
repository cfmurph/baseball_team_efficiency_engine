from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    has_name_collisions,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max_from_years,
)


def test_slider_max_from_years_uses_current_year_for_empty_metrics() -> None:
    assert slider_max_from_years([], current_year=2026) == 2026


def test_slider_max_from_years_extends_to_current_calendar_year() -> None:
    assert slider_max_from_years([1990, 2016, 2024], current_year=2026) == 2026


def test_slider_max_from_years_preserves_future_data_year() -> None:
    assert slider_max_from_years([2024, 2027], current_year=2026) == 2027


def test_player_id_columns_only_show_for_duplicate_names_with_ids() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["youngch03", "youngch04", "judgeaa01"],
            "name_full": ["Chris Young", "Chris Young", "Aaron Judge"],
        }
    )

    assert has_name_collisions(players)
    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_hide_when_names_are_unique() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert not has_name_collisions(players)
    assert player_id_columns_for_name_collisions(players) == []


def test_player_id_columns_hide_when_id_column_is_missing() -> None:
    players = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert has_name_collisions(players)
    assert player_id_columns_for_name_collisions(players) == []


def test_render_plotly_chart_delegates_to_renderer() -> None:
    fig = go.Figure()
    renderer = MagicMock()
    layout = {"plot_bgcolor": "#0d1117", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, renderer, layout, height=460)

    renderer.plotly_chart.assert_called_once_with(fig, use_container_width=True)
    assert fig.layout.height == 460
    assert fig.layout.plot_bgcolor == "#0d1117"
    assert fig.layout.paper_bgcolor == "#0d1117"
