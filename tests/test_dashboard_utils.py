from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_chart,
    scale_payroll,
    slider_max,
)


def test_slider_max_uses_current_year_for_empty_metrics() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2023, 2027], current_year=2026) == 2027


def test_render_chart_applies_layout_and_calls_streamlit_renderer_once() -> None:
    fig = go.Figure()
    plotly_chart = Mock()

    render_chart(fig, plotly_chart, height=512)

    assert fig.layout.height == 512
    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.layout.paper_bgcolor == "#0d1117"
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_scale_payroll_converts_money_columns_without_mutating_input() -> None:
    raw = pd.DataFrame({
        "payroll": [120_000_000],
        "salary": [12_500_000],
        "surplus_value": [-4_000_000],
        "dead_money_share": [0.125],
    })

    display = scale_payroll(raw)

    assert raw.loc[0, "payroll"] == 120_000_000
    assert display.loc[0, "payroll"] == 120
    assert display.loc[0, "salary"] == 12.5
    assert display.loc[0, "surplus_value"] == -4
    assert display.loc[0, "dead_money_share"] == 12.5


def test_player_id_columns_only_added_for_duplicate_names_with_ids() -> None:
    collision = pd.DataFrame({
        "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
        "name_full": ["John Smith", "John Smith", "Aaron Judge"],
    })
    unique = pd.DataFrame({
        "player_id": ["judgeaa01", "colege01"],
        "name_full": ["Aaron Judge", "Gerrit Cole"],
    })
    missing_ids = collision.drop(columns=["player_id"])

    assert player_id_columns_for_name_collisions(collision) == ["player_id"]
    assert player_id_columns_for_name_collisions(unique) == []
    assert player_id_columns_for_name_collisions(missing_ids) == []
