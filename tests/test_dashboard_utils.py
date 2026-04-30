from __future__ import annotations

import pandas as pd
import pytest

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max_for_years,
)


def test_slider_max_for_years_falls_back_to_current_year_when_no_data() -> None:
    assert slider_max_for_years([], 2026) == 2026


@pytest.mark.parametrize(
    ("years", "current_year", "expected"),
    [
        ([2021, 2022, 2024], 2026, 2026),
        ([2021, 2022, 2027], 2026, 2027),
    ],
)
def test_slider_max_for_years_keeps_slider_usable_for_data_and_future_years(
    years: list[int],
    current_year: int,
    expected: int,
) -> None:
    assert slider_max_for_years(years, current_year) == expected


class RecordingFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


def test_render_plotly_chart_applies_layout_height_and_delegates_once() -> None:
    fig = RecordingFigure()
    calls = []

    def renderer(rendered_fig, **kwargs) -> None:
        calls.append((rendered_fig, kwargs))

    render_plotly_chart(fig, renderer, layout={"template": "plotly_dark"}, height=460)

    assert fig.layout_updates == [{"template": "plotly_dark"}, {"height": 460}]
    assert calls == [(fig, {"use_container_width": True})]


def test_player_id_columns_only_appear_for_distinct_players_sharing_a_name() -> None:
    df = pd.DataFrame(
        {
            "name_full": ["Chris Young", "Chris Young", "Alex Example"],
            "player_id": ["youngch03", "youngch04", "example01"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == ["player_id"]


def test_player_id_columns_stay_hidden_for_duplicate_rows_of_same_player() -> None:
    df = pd.DataFrame(
        {
            "name_full": ["Chris Young", "Chris Young"],
            "player_id": ["youngch03", "youngch03"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == []


def test_player_id_columns_stay_hidden_when_required_columns_are_absent() -> None:
    assert player_id_columns_for_name_collisions(pd.DataFrame({"name_full": ["A"]})) == []
