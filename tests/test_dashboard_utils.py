from __future__ import annotations

import pytest

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    calculate_slider_max,
    player_id_columns_for_duplicate_names,
    render_plotly_chart,
    scale_payroll_for_display,
)

pytestmark = pytest.mark.unit


class RecordingStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[go.Figure, dict[str, object]]] = []

    def plotly_chart(self, fig: go.Figure, **kwargs: object) -> None:
        self.calls.append((fig, kwargs))


def test_calculate_slider_max_uses_current_year_for_empty_artifacts() -> None:
    assert calculate_slider_max([], current_year=2026) == 2026


def test_calculate_slider_max_extends_historical_data_to_current_year() -> None:
    assert calculate_slider_max([1990, 2001, 2016], current_year=2026) == 2026


def test_calculate_slider_max_preserves_future_artifact_year() -> None:
    assert calculate_slider_max([2024, 2027], current_year=2026) == 2027


def test_scale_payroll_for_display_converts_copy_only() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [120_000_000],
            "salary": [3_500_000],
            "surplus_value": [-9_250_000],
            "dead_money_share": [0.125],
            "wins": [90],
        }
    )

    display = scale_payroll_for_display(raw)

    assert display.loc[0, "payroll"] == 120
    assert display.loc[0, "salary"] == 3.5
    assert display.loc[0, "surplus_value"] == -9.25
    assert display.loc[0, "dead_money_share"] == 12.5
    assert display.loc[0, "wins"] == 90
    assert raw.loc[0, "payroll"] == 120_000_000


def test_player_id_columns_only_for_same_name_players_with_ids() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_duplicate_names(players) == ["player_id"]


def test_player_id_columns_omitted_without_collision_or_id_column() -> None:
    unique_names = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "ohtansh01"],
            "name_full": ["Aaron Judge", "Shohei Ohtani"],
        }
    )
    missing_id = pd.DataFrame({"name_full": ["John Smith", "John Smith"]})

    assert player_id_columns_for_duplicate_names(unique_names) == []
    assert player_id_columns_for_duplicate_names(missing_id) == []


def test_player_id_columns_stay_hidden_for_duplicate_rows_of_same_player() -> None:
    same_player_repeated = pd.DataFrame(
        {
            "player_id": ["youngch03", "youngch03"],
            "name_full": ["Chris Young", "Chris Young"],
        }
    )

    assert player_id_columns_for_duplicate_names(same_player_repeated) == []


def test_render_plotly_chart_applies_layout_height_and_delegates_once() -> None:
    fig = go.Figure(data=go.Scatter(x=[1, 2], y=[3, 4]))
    recorder = RecordingStreamlit()
    layout = {"paper_bgcolor": "#0d1117", "font": {"color": "#e6edf3"}}

    render_plotly_chart(fig, recorder, layout, height=360)

    assert len(recorder.calls) == 1
    rendered_fig, kwargs = recorder.calls[0]
    assert rendered_fig is fig
    assert kwargs == {"use_container_width": True}
    assert fig.layout.height == 360
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.font.color == "#e6edf3"
