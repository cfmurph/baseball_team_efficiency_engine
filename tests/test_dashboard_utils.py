from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    has_duplicate_player_names,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
    slider_max_year,
)


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


def test_slider_max_year_uses_current_year_when_all_years_empty():
    assert slider_max_year([], current_year=2026) == 2026


def test_slider_max_year_uses_later_data_year_or_current_year():
    assert slider_max_year([2021, 2024], current_year=2026) == 2026
    assert slider_max_year([2021, 2028], current_year=2026) == 2028


def test_player_id_columns_hidden_when_names_are_unique():
    players = pd.DataFrame({
        "player_id": ["judgeaa01", "ohtansh01"],
        "name_full": ["Aaron Judge", "Shohei Ohtani"],
    })

    assert not has_duplicate_player_names(players)
    assert player_id_columns_for_name_collisions(players) == []


def test_player_id_columns_visible_for_same_name_collision():
    players = pd.DataFrame({
        "player_id": ["gonzaal01", "gonzaal02"],
        "name_full": ["Alex Gonzalez", "Alex Gonzalez"],
        "team_name": ["Boston Red Sox", "Boston Red Sox"],
    })

    assert has_duplicate_player_names(players)
    assert player_id_columns_for_name_collisions(players) == ["player_id"]


def test_player_id_columns_hidden_when_collision_has_no_player_id():
    players = pd.DataFrame({
        "name_full": ["Alex Gonzalez", "Alex Gonzalez"],
        "team_name": ["Boston Red Sox", "Toronto Blue Jays"],
    })

    assert has_duplicate_player_names(players)
    assert player_id_columns_for_name_collisions(players) == []


def test_render_plotly_chart_applies_layout_and_calls_streamlit_once():
    fig = FakeFigure()
    st = MagicMock()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, st, layout, height=321)

    assert fig.layout_updates == [
        {"template": "plotly_dark", "paper_bgcolor": "#0d1117"},
        {"height": 321},
    ]
    st.plotly_chart.assert_called_once_with(fig, use_container_width=True)
