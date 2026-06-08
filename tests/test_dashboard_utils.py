from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    id_columns_for_name_collisions,
    render_chart,
    scale_payroll,
    slider_max,
)


def test_slider_max_uses_current_year_when_artifacts_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_preserves_future_artifact_years() -> None:
    assert slider_max([2024, 2027], current_year=2026) == 2027


def test_render_chart_applies_layout_height_and_delegates_to_streamlit() -> None:
    streamlit = Mock()
    fig = go.Figure()

    render_chart(streamlit, fig, height=525)

    assert fig.layout.height == 525
    assert fig.layout.paper_bgcolor == "#0d1117"
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_id_columns_for_name_collisions_only_when_needed() -> None:
    same_names = pd.DataFrame(
        {
            "player_id": ["smith-a", "smith-b"],
            "name_full": ["Chris Smith", "Chris Smith"],
        }
    )
    unique_names = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert id_columns_for_name_collisions(same_names) == ["player_id"]
    assert id_columns_for_name_collisions(unique_names) == []


def test_scale_payroll_converts_money_columns_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "salary": [12_500_000],
            "dead_money_share": [0.25],
            "wins": [90],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 100
    assert scaled.loc[0, "salary"] == 12.5
    assert scaled.loc[0, "dead_money_share"] == 25
    assert scaled.loc[0, "wins"] == 90
    assert raw.loc[0, "payroll"] == 100_000_000
