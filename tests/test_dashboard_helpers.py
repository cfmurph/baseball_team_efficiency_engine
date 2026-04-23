from __future__ import annotations

from unittest.mock import Mock

from src.baseball_analytics.dashboard_helpers import compute_slider_max, render_chart


def test_compute_slider_max_uses_latest_metric_year_when_newer_than_today() -> None:
    assert compute_slider_max([2021, 2023], current_year=2022) == 2023


def test_compute_slider_max_falls_back_to_current_year_for_empty_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_render_chart_applies_layout_sets_height_and_renders_once() -> None:
    fig = Mock()
    apply_layout = Mock()
    render = Mock()

    render_chart(fig, apply_layout=apply_layout, render=render, height=480)

    apply_layout.assert_called_once_with(fig)
    fig.update_layout.assert_called_once_with(height=480)
    render.assert_called_once_with(fig, use_container_width=True)
