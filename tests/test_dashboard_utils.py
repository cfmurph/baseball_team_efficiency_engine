from __future__ import annotations

from src.baseball_analytics.dashboard_utils import (
    PLOTLY_LAYOUT,
    compute_slider_max,
    render_plotly_chart,
)


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


class FakeStreamlit:
    def __init__(self) -> None:
        self.plotly_chart_calls: list[dict] = []

    def plotly_chart(self, fig, **kwargs) -> None:
        self.plotly_chart_calls.append({"fig": fig, "kwargs": kwargs})


def test_compute_slider_max_falls_back_to_current_year_when_no_metric_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_uses_latest_metric_or_current_year() -> None:
    assert compute_slider_max([1990, 1995, 2016], current_year=2026) == 2026
    assert compute_slider_max([1990, 2027, 2026], current_year=2026) == 2027


def test_render_plotly_chart_applies_theme_height_and_delegates_to_streamlit() -> None:
    fig = FakeFigure()
    st = FakeStreamlit()

    render_plotly_chart(fig, st, height=375)

    assert fig.layout_updates == [PLOTLY_LAYOUT, {"height": 375}]
    assert st.plotly_chart_calls == [
        {"fig": fig, "kwargs": {"use_container_width": True}},
    ]
