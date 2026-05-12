from __future__ import annotations

import datetime as dt
from unittest.mock import Mock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
)


def test_compute_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert compute_slider_max([], today=dt.date(2026, 5, 12)) == 2026


def test_compute_slider_max_extends_stale_metrics_to_current_year() -> None:
    assert compute_slider_max([1990, 2016], today=dt.date(2026, 5, 12)) == 2026


def test_compute_slider_max_allows_future_metric_year() -> None:
    assert compute_slider_max([2024, 2027], today=dt.date(2026, 5, 12)) == 2027


def test_render_plotly_chart_delegates_to_streamlit_plotly_chart() -> None:
    streamlit = Mock()
    fig = Mock()
    layout = {"template": "plotly_dark"}

    render_plotly_chart(streamlit, fig, height=460, layout=layout)

    fig.update_layout.assert_any_call(**layout)
    fig.update_layout.assert_any_call(height=460)
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_are_shown_only_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smitha01", "smithb01", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_are_hidden_without_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == []
