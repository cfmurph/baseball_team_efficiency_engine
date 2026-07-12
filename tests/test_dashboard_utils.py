from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    calculate_slider_max,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
)


def test_calculate_slider_max_uses_current_year_for_empty_data() -> None:
    assert calculate_slider_max([], current_year=2026) == 2026


def test_calculate_slider_max_extends_past_latest_data_year() -> None:
    assert calculate_slider_max([1990, 2016, 2024], current_year=2026) == 2026


def test_calculate_slider_max_preserves_future_data_year() -> None:
    assert calculate_slider_max([2027], current_year=2026) == 2027


def test_player_id_column_only_shown_for_same_name_collisions() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["smithjo01", "smithjo02", "judgeaa01"],
            "name_full": ["John Smith", "John Smith", "Aaron Judge"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == ["player_id"]


def test_player_id_column_hidden_without_collisions() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["judgeaa01", "colege01"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )

    assert player_id_columns_for_name_collisions(df) == []


def test_player_id_column_hidden_when_id_is_unavailable() -> None:
    df = pd.DataFrame({"name_full": ["John Smith", "John Smith"]})

    assert player_id_columns_for_name_collisions(df) == []


def test_scale_payroll_returns_scaled_copy() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [200_000_000],
            "salary": [30_000_000],
            "dead_money_share": [0.25],
            "wins": [95],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 200
    assert scaled.loc[0, "salary"] == 30
    assert scaled.loc[0, "dead_money_share"] == 25
    assert raw.loc[0, "payroll"] == 200_000_000


def test_render_plotly_chart_delegates_to_streamlit_plotly_chart() -> None:
    fig = MagicMock()
    st = MagicMock()
    layout = {"template": "plotly_dark"}

    render_plotly_chart(fig, st, layout, height=460)

    fig.update_layout.assert_any_call(template="plotly_dark")
    fig.update_layout.assert_any_call(height=460)
    st.plotly_chart.assert_called_once_with(fig, use_container_width=True)
