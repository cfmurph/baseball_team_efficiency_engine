from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    has_name_collision,
    player_id_disambiguation_columns,
    render_plotly_chart,
)


def test_compute_slider_max_includes_current_year_when_data_lags() -> None:
    assert compute_slider_max([2021, 2023, 2022], current_year=2026) == 2026


def test_compute_slider_max_handles_empty_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_preserves_future_data_year() -> None:
    assert compute_slider_max([2024, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.layout_updates: list[dict] = []

        def update_layout(self, **kwargs) -> None:
            self.layout_updates.append(kwargs)

    class FakeStreamlit:
        def __init__(self) -> None:
            self.chart_calls: list[tuple[FakeFigure, dict]] = []

        def plotly_chart(self, fig, **kwargs) -> None:
            self.chart_calls.append((fig, kwargs))

    fig = FakeFigure()
    streamlit_api = FakeStreamlit()

    render_plotly_chart(fig, streamlit_api, {"template": "plotly_dark"}, height=525)

    assert fig.layout_updates == [{"template": "plotly_dark"}, {"height": 525}]
    assert streamlit_api.chart_calls == [(fig, {"use_container_width": True})]


def test_player_id_disambiguation_columns_prepend_id_for_same_name_players() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["griffke02", "grifffe01"],
            "name_full": ["Ken Griffey", "Ken Griffey"],
        }
    )

    assert has_name_collision(df)
    assert player_id_disambiguation_columns(df, ["name_full", "team_name"]) == [
        "player_id",
        "name_full",
        "team_name",
    ]


def test_player_id_disambiguation_columns_leave_unique_names_unchanged() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert not has_name_collision(df)
    assert player_id_disambiguation_columns(df, ["name_full", "team_name"]) == [
        "name_full",
        "team_name",
    ]
