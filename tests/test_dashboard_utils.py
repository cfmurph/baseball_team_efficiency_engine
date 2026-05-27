from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll_columns,
    slider_max_year,
)


def test_scale_payroll_columns_converts_display_units_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [120_000_000.0],
            "salary": [12_500_000.0],
            "cost_per_war": [8_000_000.0],
            "surplus_value": [-4_000_000.0],
            "dead_money_share": [0.125],
            "wins": [90],
        }
    )

    scaled = scale_payroll_columns(raw)

    assert scaled.loc[0, "payroll"] == pytest.approx(120.0)
    assert scaled.loc[0, "salary"] == pytest.approx(12.5)
    assert scaled.loc[0, "cost_per_war"] == pytest.approx(8.0)
    assert scaled.loc[0, "surplus_value"] == pytest.approx(-4.0)
    assert scaled.loc[0, "dead_money_share"] == pytest.approx(12.5)
    assert scaled.loc[0, "wins"] == 90
    assert raw.loc[0, "payroll"] == pytest.approx(120_000_000.0)


def test_slider_max_year_handles_empty_metrics_years() -> None:
    assert slider_max_year([], current_year=2026) == 2026


def test_slider_max_year_extends_to_current_year_when_artifacts_are_stale() -> None:
    assert slider_max_year([2022, 2024], current_year=2026) == 2026
    assert slider_max_year([2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_and_calls_streamlit_once() -> None:
    fig = go.Figure()
    streamlit = Mock()

    render_plotly_chart(fig, streamlit, {"paper_bgcolor": "#000000"}, height=333)

    assert fig.layout.paper_bgcolor == "#000000"
    assert fig.layout.height == 333
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_only_when_names_collide_and_ids_available() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["a", "b", "c"],
            "name_full": ["Chris Young", "Chris Young", "Other Player"],
        }
    )
    no_collision = pd.DataFrame(
        {
            "player_id": ["a", "b"],
            "name_full": ["Chris Young", "Other Player"],
        }
    )
    missing_ids = players.drop(columns=["player_id"])

    assert player_id_columns_for_name_collisions(players) == ["player_id"]
    assert player_id_columns_for_name_collisions(no_collision) == []
    assert player_id_columns_for_name_collisions(missing_ids) == []
