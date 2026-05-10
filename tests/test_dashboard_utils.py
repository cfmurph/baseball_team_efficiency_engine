from __future__ import annotations

from unittest.mock import Mock

import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import compute_slider_max, render_plotly_chart


def test_compute_slider_max_uses_current_year_when_no_artifact_years() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_keeps_future_artifact_year() -> None:
    assert compute_slider_max([2020, 2024, 2027], 2026) == 2027


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    st = Mock()
    fig = go.Figure()

    render_plotly_chart(st, fig, {"paper_bgcolor": "#0d1117"}, height=460)

    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.height == 460
    st.plotly_chart.assert_called_once_with(fig, use_container_width=True)
