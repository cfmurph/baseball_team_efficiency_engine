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


def test_slider_max_uses_current_year_for_empty_or_stale_data() -> None:
    assert slider_max([], current_year=2026) == 2026
    assert slider_max([2018, 2024], current_year=2026) == 2026


def test_slider_max_keeps_future_metric_year_selectable() -> None:
    assert slider_max([2024, 2027], current_year=2026) == 2027


def test_render_plotly_chart_applies_layout_height_and_delegates_to_streamlit() -> None:
    fig = go.Figure()
    plotly_chart = Mock()

    render_plotly_chart(fig, plotly_chart, height=512)

    assert fig.layout.height == 512
    assert fig.layout.paper_bgcolor == "#0d1117"
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_scale_payroll_converts_money_columns_without_mutating_input() -> None:
    source = pd.DataFrame(
        {
            "payroll": [120_000_000],
            "salary": [5_500_000],
            "surplus_value": [-2_000_000],
            "dead_money_share": [0.125],
            "wins": [90],
        }
    )

    scaled = scale_payroll(source)

    assert scaled.loc[0, "payroll"] == 120
    assert scaled.loc[0, "salary"] == 5.5
    assert scaled.loc[0, "surplus_value"] == -2
    assert scaled.loc[0, "dead_money_share"] == 12.5
    assert source.loc[0, "payroll"] == 120_000_000


def test_player_id_columns_only_shown_for_same_name_collisions() -> None:
    collisions = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )
    unique_names = collisions.drop(index=1)

    assert player_id_columns_for_name_collisions(collisions) == ["player_id"]
    assert player_id_columns_for_name_collisions(unique_names) == []
    assert player_id_columns_for_name_collisions(collisions.drop(columns=["player_id"])) == []
