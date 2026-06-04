from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.baseball_analytics.dashboard_utils import (
    render_plotly_chart,
    scale_payroll_for_display,
    slider_max_year,
)


def test_slider_max_year_handles_empty_metrics() -> None:
    assert slider_max_year([], current_year=2026) == 2026


def test_slider_max_year_keeps_future_metric_year_selectable() -> None:
    assert slider_max_year([1998, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_height_and_delegates_once() -> None:
    fig = go.Figure()
    calls = []

    def fake_renderer(rendered_fig, **kwargs) -> None:
        calls.append((rendered_fig, kwargs))

    render_plotly_chart(fig, fake_renderer, height=512)

    assert calls == [(fig, {"use_container_width": True})]
    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"


def test_scale_payroll_for_display_converts_raw_dollars_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [120_000_000.0],
            "salary": [12_500_000.0],
            "surplus_value": [-8_000_000.0],
            "dead_money_share": [0.125],
        }
    )

    scaled = scale_payroll_for_display(raw)

    assert scaled.loc[0, "payroll"] == pytest.approx(120.0)
    assert scaled.loc[0, "salary"] == pytest.approx(12.5)
    assert scaled.loc[0, "surplus_value"] == pytest.approx(-8.0)
    assert scaled.loc[0, "dead_money_share"] == pytest.approx(12.5)
    assert raw.loc[0, "payroll"] == 120_000_000.0
