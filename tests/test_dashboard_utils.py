from __future__ import annotations

from unittest.mock import Mock

from src.baseball_analytics.dashboard_utils import compute_slider_max, render_chart


def test_compute_slider_max_uses_current_year_when_no_data() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_uses_latest_data_year_when_newer_than_current() -> None:
    assert compute_slider_max([2019, 2020, 2027], 2026) == 2027


def test_compute_slider_max_uses_current_year_when_newer_than_data() -> None:
    assert compute_slider_max([2019, 2020, 2024], 2026) == 2026


def test_render_chart_applies_layout_and_renders_once() -> None:
    fig = Mock()
    apply_layout = Mock()
    plotly_chart = Mock()

    render_chart(fig, apply_layout, plotly_chart, height=480)

    apply_layout.assert_called_once_with(fig)
    fig.update_layout.assert_called_once_with(height=480)
    plotly_chart.assert_called_once_with(fig, use_container_width=True)
