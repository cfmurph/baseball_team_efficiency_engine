from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll_columns,
    slider_max,
)


def test_slider_max_handles_empty_and_stale_years() -> None:
    assert slider_max([], current_year=2026) == 2026
    assert slider_max([1990, 2016], current_year=2026) == 2026
    assert slider_max([2024, 2028], current_year=2026) == 2028


def test_player_id_column_only_added_for_visible_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["griffke02", "griffke01", "judgeaa01"],
            "name_full": ["Ken Griffey", "Ken Griffey", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == ["player_id"]
    assert player_id_columns_for_name_collisions(players.drop(columns=["player_id"])) == []
    assert player_id_columns_for_name_collisions(players.iloc[[0, 2]]) == []


def test_scale_payroll_columns_copies_and_scales_display_values() -> None:
    source = pd.DataFrame(
        {
            "payroll": [120_000_000],
            "salary": [30_000_000],
            "surplus_value": [-12_500_000],
            "dead_money_share": [0.125],
        }
    )

    result = scale_payroll_columns(source)

    assert result.loc[0, "payroll"] == 120
    assert result.loc[0, "salary"] == 30
    assert result.loc[0, "surplus_value"] == -12.5
    assert result.loc[0, "dead_money_share"] == 12.5
    assert source.loc[0, "payroll"] == 120_000_000


def test_render_plotly_chart_delegates_to_streamlit_plotly_chart_once() -> None:
    fig = go.Figure()
    streamlit = SimpleNamespace(plotly_chart=Mock())

    render_plotly_chart(streamlit, fig, {"paper_bgcolor": "#0d1117"}, height=321)

    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.height == 321
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
