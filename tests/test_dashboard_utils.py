from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_collisions,
    render_plotly_chart,
    slider_max_year,
)


class DummyFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


def test_slider_max_year_uses_current_year_when_metrics_are_empty() -> None:
    assert slider_max_year([], 2026) == 2026


def test_slider_max_year_preserves_future_artifact_year() -> None:
    assert slider_max_year([2024, 2027], 2026) == 2027


def test_render_plotly_chart_applies_layout_height_and_delegates_once() -> None:
    fig = DummyFigure()
    calls = []

    def renderer(rendered_fig, **kwargs) -> None:
        calls.append((rendered_fig, kwargs))

    render_plotly_chart(fig, renderer, height=460)

    assert fig.layout_updates[0]["template"] == "plotly_dark"
    assert fig.layout_updates[-1] == {"height": 460}
    assert calls == [(fig, {"use_container_width": True})]


def test_player_id_column_is_added_only_for_same_name_collisions() -> None:
    collision = pd.DataFrame(
        {
            "player_id": ["same01", "same02", "other01"],
            "name_full": ["Alex Smith", "Alex Smith", "Jordan Lee"],
        }
    )
    no_collision = pd.DataFrame(
        {
            "player_id": ["same01", "other01"],
            "name_full": ["Alex Smith", "Jordan Lee"],
        }
    )
    no_id = pd.DataFrame({"name_full": ["Alex Smith", "Alex Smith"]})

    assert player_id_columns_for_collisions(collision) == ["player_id"]
    assert player_id_columns_for_collisions(no_collision) == []
    assert player_id_columns_for_collisions(no_id) == []
