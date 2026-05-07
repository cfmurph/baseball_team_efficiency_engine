from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    columns_with_player_id_for_collisions,
    compute_slider_max,
    has_player_name_collision,
    render_plotly_chart,
)


def test_compute_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_keeps_future_artifact_year_available() -> None:
    assert compute_slider_max([2022, 2027], 2026) == 2027


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = go.Figure()
    streamlit = Mock()

    render_plotly_chart(
        fig,
        streamlit,
        {"template": "plotly_dark", "paper_bgcolor": "#0d1117"},
        height=275,
    )

    assert fig.layout.height == 275
    assert fig.layout.paper_bgcolor == "#0d1117"
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_player_id_columns_are_prefixed_only_for_same_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smith001", "smith002", "judge001"],
            "name_full": ["Alex Smith", "Alex Smith", "Aaron Judge"],
            "player_war": [2.1, 0.4, 8.3],
        }
    )

    assert has_player_name_collision(players)
    assert columns_with_player_id_for_collisions(
        players,
        ["name_full", "player_war"],
    ) == ["player_id", "name_full", "player_war"]


def test_player_id_columns_do_not_show_without_collision_or_id() -> None:
    unique_players = pd.DataFrame(
        {
            "player_id": ["judge001", "cole001"],
            "name_full": ["Aaron Judge", "Gerrit Cole"],
        }
    )
    missing_ids = pd.DataFrame({"name_full": ["Alex Smith", "Alex Smith"]})

    assert not has_player_name_collision(unique_players)
    assert columns_with_player_id_for_collisions(
        unique_players,
        ["name_full"],
    ) == ["name_full"]
    assert columns_with_player_id_for_collisions(missing_ids, ["name_full"]) == ["name_full"]
