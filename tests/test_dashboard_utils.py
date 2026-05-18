from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    player_id_prefix_columns,
    render_plotly_chart,
)


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


def test_compute_slider_max_uses_current_year_when_metrics_are_empty():
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_extends_past_latest_metric_year():
    assert compute_slider_max([2021, 2024], current_year=2026) == 2026
    assert compute_slider_max([2021, 2028], current_year=2026) == 2028


def test_player_id_prefix_columns_only_when_name_collision_needs_disambiguation():
    distinct = pd.DataFrame(
        {
            "player_id": ["griffke02", "griffke01"],
            "name_full": ["Ken Griffey Jr.", "Ken Griffey Sr."],
        }
    )
    same_name = pd.DataFrame(
        {
            "player_id": ["gonzale01", "gonzale02"],
            "name_full": ["Alex Gonzalez", "Alex Gonzalez"],
        }
    )

    assert player_id_prefix_columns(distinct) == []
    assert player_id_prefix_columns(same_name) == ["player_id"]


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit_once():
    fig = FakeFigure()
    calls = []

    def plotly_chart(rendered_fig, **kwargs):
        calls.append((rendered_fig, kwargs))

    render_plotly_chart(fig, plotly_chart, {"template": "plotly_dark"}, height=320)

    assert fig.layout_updates == [{"template": "plotly_dark"}, {"height": 320}]
    assert calls == [(fig, {"use_container_width": True})]
