from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.baseball_analytics.dashboard_utils import render_plotly_chart, slider_max


@pytest.mark.parametrize(
    ("all_years", "current_year", "expected"),
    [
        ([], 2026, 2026),
        ([1990, 2001, 2024], 2026, 2026),
        ([1990, 2030], 2026, 2030),
    ],
)
def test_slider_max_handles_empty_and_future_metric_years(
    all_years: list[int],
    current_year: int,
    expected: int,
) -> None:
    assert slider_max(all_years, current_year) == expected


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict[str, object]] = []

    def update_layout(self, **kwargs: object) -> None:
        self.layout_updates.append(kwargs)


def test_render_plotly_chart_applies_layout_height_and_calls_streamlit() -> None:
    fig = FakeFigure()
    streamlit = Mock()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, streamlit, layout, height=460)

    assert fig.layout_updates == [
        {"template": "plotly_dark", "paper_bgcolor": "#0d1117"},
        {"height": 460},
    ]
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
