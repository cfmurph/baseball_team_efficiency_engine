from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    slider_max,
)


def test_slider_max_uses_current_year_when_no_artifact_years() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_artifact_year_selectable() -> None:
    assert slider_max([2024, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = go.Figure()
    plotly_chart = Mock()

    render_plotly_chart(fig, plotly_chart, height=525)

    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.height == 525
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_only_added_for_same_name_collisions() -> None:
    same_name = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02"],
            "name_full": ["John Smith", "John Smith"],
        }
    )
    distinct_names = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert player_id_columns_for_name_collisions(same_name) == ["player_id"]
    assert player_id_columns_for_name_collisions(distinct_names) == []


def test_scale_payroll_converts_display_units_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [200_000_000.0],
            "salary": [35_000_000.0],
            "surplus_value": [-12_500_000.0],
            "dead_money_share": [0.075],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 200.0
    assert scaled.loc[0, "salary"] == 35.0
    assert scaled.loc[0, "surplus_value"] == -12.5
    assert scaled.loc[0, "dead_money_share"] == 7.5
    assert raw.loc[0, "payroll"] == 200_000_000.0
