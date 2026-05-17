from __future__ import annotations

from unittest.mock import Mock

from src.baseball_analytics.dashboard_utils import compute_slider_max, render_plotly_chart


def test_compute_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_includes_future_or_current_bounds() -> None:
    assert compute_slider_max([2021, 2024], current_year=2026) == 2026
    assert compute_slider_max([2021, 2028], current_year=2026) == 2028


def test_render_plotly_chart_delegates_to_streamlit_plotly_chart() -> None:
    streamlit_api = Mock()
    fig = Mock()
    apply_layout = Mock()

    render_plotly_chart(streamlit_api, fig, height=512, apply_layout=apply_layout)

    apply_layout.assert_called_once_with(fig)
    fig.update_layout.assert_called_once_with(height=512)
    streamlit_api.plotly_chart.assert_called_once_with(fig, use_container_width=True)
