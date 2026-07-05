from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    has_name_collision,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    slider_max,
)


def test_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_keeps_future_artifact_years_selectable() -> None:
    assert slider_max([2023, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_and_delegates_once() -> None:
    fig = go.Figure()
    st = MagicMock()

    render_plotly_chart(fig, st, height=325)

    assert fig.layout.template is not None
    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.height == 325
    st.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_column_is_added_only_for_same_name_collisions() -> None:
    distinct = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )
    collision = pd.DataFrame(
        {
            "player_id": ["garcilu01", "garcilu02"],
            "name_full": ["Luis Garcia", "Luis Garcia"],
        }
    )

    assert not has_name_collision(distinct)
    assert player_id_columns_for_name_collisions(distinct) == []
    assert has_name_collision(collision)
    assert player_id_columns_for_name_collisions(collision) == ["player_id"]


def test_player_id_column_is_omitted_when_collision_has_no_id_column() -> None:
    players = pd.DataFrame({"name_full": ["Luis Garcia", "Luis Garcia"]})

    assert has_name_collision(players)
    assert player_id_columns_for_name_collisions(players) == []


def test_scale_payroll_converts_money_columns_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "salary": [5_000_000],
            "surplus_value": [-2_500_000],
            "dead_money_share": [0.125],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 100
    assert scaled.loc[0, "salary"] == 5
    assert scaled.loc[0, "surplus_value"] == -2.5
    assert scaled.loc[0, "dead_money_share"] == 12.5
    assert raw.loc[0, "payroll"] == 100_000_000
