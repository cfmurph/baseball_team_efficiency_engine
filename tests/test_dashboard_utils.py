from __future__ import annotations

import datetime
from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    slider_max_for_years,
)


def test_slider_max_for_empty_years_uses_current_year() -> None:
    today = datetime.date(2026, 6, 15)

    assert slider_max_for_years([], today=today) == 2026


def test_slider_max_keeps_future_artifact_year_selectable() -> None:
    today = datetime.date(2026, 6, 15)

    assert slider_max_for_years([2024, 2027, 2025], today=today) == 2027


def test_slider_max_extends_stale_artifacts_to_current_year() -> None:
    today = datetime.date(2026, 6, 15)

    assert slider_max_for_years([1990, 2016, 2024], today=today) == 2026


def test_render_plotly_chart_applies_layout_and_delegates_to_renderer() -> None:
    fig = go.Figure()
    renderer = Mock()

    render_plotly_chart(fig, renderer, height=512)

    renderer.assert_called_once_with(fig, use_container_width=True)
    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"


def test_player_id_column_shown_for_same_name_players() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == ["player_id"]


def test_player_id_column_hidden_without_name_collisions() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["smithjo01", "judgeaa01"],
            "name_full": ["John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == []


def test_player_id_column_hidden_when_id_is_unavailable() -> None:
    df = pd.DataFrame({"name_full": ["John Smith", "John Smith"]})

    assert player_id_columns_for_name_collisions(df) == []


def test_scale_payroll_converts_display_columns_without_mutating_input() -> None:
    df = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "salary": [12_500_000],
            "dead_money_share": [0.25],
            "wins": [90],
        }
    )

    result = scale_payroll(df)

    assert result.loc[0, "payroll"] == 100
    assert result.loc[0, "salary"] == 12.5
    assert result.loc[0, "dead_money_share"] == 25
    assert result.loc[0, "wins"] == 90
    assert df.loc[0, "payroll"] == 100_000_000
