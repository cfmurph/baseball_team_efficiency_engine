from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    has_player_name_collision,
    player_id_columns,
    render_plotly_chart,
    slider_max,
)


class _FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.plotly_calls: list[tuple[object, dict]] = []

    def plotly_chart(self, fig, **kwargs) -> None:
        self.plotly_calls.append((fig, kwargs))


def test_slider_max_uses_current_year_for_empty_metrics() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_data_selectable() -> None:
    assert slider_max([2024, 2030, 2029], current_year=2026) == 2030


def test_render_plotly_chart_styles_then_delegates_once() -> None:
    fig = _FakeFigure()
    st = _FakeStreamlit()

    render_plotly_chart(fig, st, height=525)

    assert fig.layout_updates[0]["template"] == "plotly_dark"
    assert fig.layout_updates[1] == {"height": 525}
    assert st.plotly_calls == [(fig, {"use_container_width": True})]


def test_player_id_columns_added_for_same_name_players() -> None:
    df = pd.DataFrame(
        {
            "name_full": ["Chris Young", "Chris Young", "Aaron Judge"],
            "player_id": ["youngch03", "youngch04", "judgeaa01"],
        }
    )

    assert player_id_columns(df) == ["player_id"]


def test_name_collision_detected_even_without_player_id() -> None:
    df = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert has_player_name_collision(df)


def test_player_id_columns_omitted_without_disambiguating_id() -> None:
    df = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert player_id_columns(df) == []
