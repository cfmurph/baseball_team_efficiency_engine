from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    collision_id_columns,
    has_name_collision,
    render_plotly_chart,
    slider_max,
)


def test_slider_max_uses_current_year_when_no_metric_years() -> None:
    assert slider_max([], 2026) == 2026


def test_slider_max_includes_future_data_year() -> None:
    assert slider_max([2024, 2027], 2026) == 2027


def test_collision_id_columns_only_when_same_name_players_can_be_disambiguated() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["griffke01", "griffke02"],
            "name_full": ["Ken Griffey", "Ken Griffey"],
        }
    )

    assert has_name_collision(players)
    assert collision_id_columns(players) == ["player_id"]


def test_collision_id_columns_omits_id_without_name_collision() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["griffke02", "jeterde01"],
            "name_full": ["Ken Griffey", "Derek Jeter"],
        }
    )

    assert not has_name_collision(players)
    assert collision_id_columns(players) == []


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = Mock()
    plotly_chart = Mock(return_value="rendered")

    result = render_plotly_chart(
        fig,
        plotly_chart,
        {"template": "plotly_dark"},
        height=320,
    )

    assert result == "rendered"
    fig.update_layout.assert_any_call(template="plotly_dark")
    fig.update_layout.assert_any_call(height=320)
    plotly_chart.assert_called_once_with(fig, use_container_width=True)
