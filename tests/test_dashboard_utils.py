from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    has_name_collisions,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
)


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


class FakeStreamlit:
    def __init__(self) -> None:
        self.plotly_calls: list[tuple[object, dict]] = []

    def plotly_chart(self, fig, **kwargs) -> None:
        self.plotly_calls.append((fig, kwargs))


def test_compute_slider_max_uses_current_year_when_metric_years_empty() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_extends_past_latest_metric_year_to_current_year() -> None:
    assert compute_slider_max([1990, 2016], current_year=2026) == 2026


def test_compute_slider_max_keeps_future_metric_year_available() -> None:
    assert compute_slider_max([2024, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_height_and_delegates_to_streamlit() -> None:
    fig = FakeFigure()
    st = FakeStreamlit()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, st, layout, height=320)

    assert fig.layout_updates == [layout, {"height": 320}]
    assert st.plotly_calls == [(fig, {"use_container_width": True})]


def test_player_id_column_is_added_only_for_same_name_collisions() -> None:
    players = pd.DataFrame({
        "player_id": ["smith001", "smith002", "judge001"],
        "name_full": ["Chris Smith", "Chris Smith", "Aaron Judge"],
    })

    assert has_name_collisions(players)
    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_column_is_omitted_without_ambiguous_names() -> None:
    players = pd.DataFrame({
        "player_id": ["judge001", "cole001"],
        "name_full": ["Aaron Judge", "Gerrit Cole"],
    })

    assert not has_name_collisions(players)
    assert player_id_columns_for_name_collisions(players) == []
