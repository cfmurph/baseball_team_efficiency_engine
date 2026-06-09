from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    PLOTLY_LAYOUT,
    has_name_collision,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max,
)


class RecordingFigure:
    def __init__(self) -> None:
        self.layout_calls: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_calls.append(kwargs)


def test_slider_max_uses_current_year_when_metric_years_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2024, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_and_calls_streamlit_renderer() -> None:
    fig = RecordingFigure()
    render_calls = []

    def renderer(rendered_fig, **kwargs) -> None:
        render_calls.append((rendered_fig, kwargs))

    render_plotly_chart(fig, renderer, height=460)

    assert fig.layout_calls[0] == PLOTLY_LAYOUT
    assert fig.layout_calls[1] == {"height": 460}
    assert render_calls == [(fig, {"use_container_width": True})]


def test_player_id_column_is_added_only_for_same_name_players() -> None:
    same_name = pd.DataFrame(
        {
            "player_id": ["griffke01", "grifffe01"],
            "name_full": ["Ken Griffey", "Ken Griffey"],
        }
    )
    unique_names = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "ohtansh01"],
            "name_full": ["Aaron Judge", "Shohei Ohtani"],
        }
    )

    assert has_name_collision(same_name)
    assert player_id_columns_for_name_collisions(same_name) == ["player_id"]
    assert not has_name_collision(unique_names)
    assert player_id_columns_for_name_collisions(unique_names) == []


def test_player_id_column_is_not_added_when_id_is_unavailable() -> None:
    df = pd.DataFrame({"name_full": ["Alex Gonzalez", "Alex Gonzalez"]})

    assert has_name_collision(df)
    assert player_id_columns_for_name_collisions(df) == []
