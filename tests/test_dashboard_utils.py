from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max,
)


def test_slider_max_uses_current_year_when_artifacts_have_no_years() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_allows_future_artifact_year() -> None:
    assert slider_max([2024, 2027], current_year=2026) == 2027


def test_player_id_columns_included_only_for_same_name_players() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smith001", "smith002", "jones001"],
            "name_full": ["Alex Smith", "Alex Smith", "Sam Jones"],
        }
    )

    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_omitted_without_collision_or_id() -> None:
    unique_names = pd.DataFrame(
        {
            "player_id": ["smith001", "jones001"],
            "name_full": ["Alex Smith", "Sam Jones"],
        }
    )
    missing_id = pd.DataFrame({"name_full": ["Alex Smith", "Alex Smith"]})

    assert player_id_columns_for_name_collisions(unique_names) == []
    assert player_id_columns_for_name_collisions(missing_id) == []


def test_render_plotly_chart_applies_layout_once_then_calls_streamlit_renderer() -> None:
    fig = Mock()
    st = Mock()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, st, layout, height=460)

    assert fig.update_layout.call_args_list[0].kwargs == layout
    assert fig.update_layout.call_args_list[1].kwargs == {"height": 460}
    st.plotly_chart.assert_called_once_with(fig, use_container_width=True)
