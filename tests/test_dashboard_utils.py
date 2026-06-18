from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns,
    render_plotly_chart,
    slider_max,
)


class FakeStreamlit:
    def __init__(self) -> None:
        self.plotly_chart_calls = []

    def plotly_chart(self, fig, **kwargs) -> None:
        self.plotly_chart_calls.append((fig, kwargs))


def test_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2024, 2028, 2026], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_and_delegates_once() -> None:
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    fake_st = FakeStreamlit()

    render_plotly_chart(fig, fake_st, height=512)

    assert len(fake_st.plotly_chart_calls) == 1
    rendered_fig, kwargs = fake_st.plotly_chart_calls[0]
    assert rendered_fig is fig
    assert kwargs == {"use_container_width": True}
    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"


def test_player_id_columns_shows_id_only_for_same_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smitha01", "smithb01", "jones01"],
            "name_full": ["Alex Smith", "Alex Smith", "Jordan Jones"],
        }
    )

    assert player_id_columns(players) == ["player_id"]


def test_player_id_columns_omits_id_without_collision() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smitha01", "jones01"],
            "name_full": ["Alex Smith", "Jordan Jones"],
        }
    )

    assert player_id_columns(players) == []


def test_player_id_columns_omits_missing_id_column() -> None:
    players = pd.DataFrame({"name_full": ["Alex Smith", "Alex Smith"]})

    assert player_id_columns(players) == []
