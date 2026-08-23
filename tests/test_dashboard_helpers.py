from __future__ import annotations

from unittest.mock import Mock

from src.baseball_analytics.dashboard_helpers import (
    apply_layout_and_render_chart,
    compute_slider_max,
)


def test_compute_slider_max_uses_current_year_when_no_data() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_uses_latest_metric_year_when_greater() -> None:
    assert compute_slider_max([2018, 2019, 2020], 2017) == 2020


def test_compute_slider_max_uses_current_year_when_greater() -> None:
    assert compute_slider_max([2018, 2019, 2020], 2026) == 2026


def test_apply_layout_and_render_chart_applies_layout_then_renders() -> None:
    events: list[str] = []

    class DummyFigure:
        def __init__(self) -> None:
            self.last_layout_kwargs: dict[str, int] | None = None

        def update_layout(self, **kwargs: int) -> None:
            events.append("update_layout")
            self.last_layout_kwargs = kwargs

    fig = DummyFigure()

    def fake_apply_layout(figure: DummyFigure) -> None:
        events.append("apply_layout")
        assert figure is fig

    plotly_chart = Mock()
    apply_layout_and_render_chart(
        fig,
        apply_layout=fake_apply_layout,
        plotly_chart=plotly_chart,
        height=460,
    )

    assert events == ["apply_layout", "update_layout"]
    assert fig.last_layout_kwargs == {"height": 460}
    plotly_chart.assert_called_once_with(fig, use_container_width=True)
