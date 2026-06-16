from __future__ import annotations

from unittest.mock import Mock

import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import render_plotly_chart, slider_max


def test_slider_max_uses_current_year_when_no_metrics_years() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metrics_year_selectable() -> None:
    assert slider_max([2024, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_height_and_delegates_once() -> None:
    fig = go.Figure()
    plotly_chart = Mock()

    render_plotly_chart(fig, plotly_chart=plotly_chart, height=512)

    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    assert fig.layout.height == 512
    plotly_chart.assert_called_once_with(fig, use_container_width=True)
