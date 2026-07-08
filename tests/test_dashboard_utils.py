from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    slider_max_year,
)


def test_slider_max_year_uses_current_year_for_empty_data() -> None:
    assert slider_max_year([], 2026) == 2026


def test_slider_max_year_extends_historical_data_to_current_year() -> None:
    assert slider_max_year([2014, 2015, 2016], 2026) == 2026


def test_slider_max_year_preserves_future_artifact_year() -> None:
    assert slider_max_year([2024, 2027], 2026) == 2027


def test_player_id_columns_only_shown_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["youngch03", "youngch04", "judgeaa01"],
            "name_full": ["Chris Young", "Chris Young", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_hidden_when_names_are_unique_or_id_missing() -> None:
    unique_names = pd.DataFrame(
        {"player_id": ["judgeaa01", "colege01"], "name_full": ["Aaron Judge", "Gerrit Cole"]}
    )
    missing_id = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert player_id_columns_for_name_collisions(unique_names) == []
    assert player_id_columns_for_name_collisions(missing_id) == []


def test_scale_payroll_converts_display_columns_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "salary": [12_500_000],
            "surplus_value": [-2_000_000],
            "dead_money_share": [0.125],
            "wins": [90],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 100
    assert scaled.loc[0, "salary"] == 12.5
    assert scaled.loc[0, "surplus_value"] == -2
    assert scaled.loc[0, "dead_money_share"] == 12.5
    assert scaled.loc[0, "wins"] == 90
    assert raw.loc[0, "payroll"] == 100_000_000


def test_render_plotly_chart_applies_layout_height_and_delegates_to_streamlit() -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.layout_calls: list[dict] = []

        def update_layout(self, **kwargs) -> None:
            self.layout_calls.append(kwargs)

    class FakeStreamlit:
        def __init__(self) -> None:
            self.calls: list[tuple[FakeFigure, bool]] = []

        def plotly_chart(self, fig: FakeFigure, use_container_width: bool) -> None:
            self.calls.append((fig, use_container_width))

    fig = FakeFigure()
    st = FakeStreamlit()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, st, layout, height=480)

    assert fig.layout_calls == [layout, {"height": 480}]
    assert st.calls == [(fig, True)]
