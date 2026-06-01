from __future__ import annotations

import datetime

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    year_slider_max,
)


def test_year_slider_max_falls_back_to_current_year_for_empty_metrics() -> None:
    today = datetime.date(2026, 6, 1)

    assert year_slider_max([], today=today) == 2026


def test_year_slider_max_extends_historical_metrics_to_current_year() -> None:
    today = datetime.date(2026, 6, 1)

    assert year_slider_max([1990, 2016], today=today) == 2026
    assert year_slider_max([1990, 2027], today=today) == 2027


def test_render_plotly_chart_applies_layout_height_and_delegates_to_streamlit() -> None:
    class FakeStreamlit:
        def __init__(self) -> None:
            self.calls = []

        def plotly_chart(self, fig, **kwargs) -> None:
            self.calls.append((fig, kwargs))

    fake_st = FakeStreamlit()
    fig = go.Figure()

    render_plotly_chart(
        fig,
        fake_st,
        {"paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117"},
        height=460,
    )

    assert len(fake_st.calls) == 1
    rendered_fig, kwargs = fake_st.calls[0]
    assert rendered_fig is fig
    assert kwargs == {"use_container_width": True}
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    assert fig.layout.height == 460


def test_player_id_columns_only_when_same_name_players_need_disambiguation() -> None:
    collision_df = pd.DataFrame(
        {
            "player_id": ["youngch01", "youngch03"],
            "name_full": ["Chris Young", "Chris Young"],
        }
    )
    unique_df = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )
    missing_id_df = pd.DataFrame({"name_full": ["Chris Young", "Chris Young"]})

    assert player_id_columns_for_name_collisions(collision_df) == ["player_id"]
    assert player_id_columns_for_name_collisions(unique_df) == []
    assert player_id_columns_for_name_collisions(missing_id_df) == []


def test_scale_payroll_converts_display_units_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [120_000_000],
            "salary": [2_500_000],
            "surplus_value": [-4_000_000],
            "dead_money_share": [0.25],
        }
    )

    display = scale_payroll(raw)

    assert display.loc[0, "payroll"] == 120
    assert display.loc[0, "salary"] == 2.5
    assert display.loc[0, "surplus_value"] == -4
    assert display.loc[0, "dead_money_share"] == 25
    assert raw.loc[0, "payroll"] == 120_000_000
