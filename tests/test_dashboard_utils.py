from __future__ import annotations

from unittest.mock import Mock

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    PLOTLY_LAYOUT,
    id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    slider_max,
)


class DummyFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


def test_slider_max_uses_current_year_when_years_are_empty() -> None:
    assert slider_max([], current_year=2026) == 2026


def test_slider_max_uses_later_of_artifact_year_and_current_year() -> None:
    assert slider_max([2023, 2024], current_year=2026) == 2026
    assert slider_max([2023, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = DummyFigure()
    plotly_chart = Mock()

    render_plotly_chart(fig, plotly_chart, height=512)

    assert fig.layout_updates[0] == PLOTLY_LAYOUT
    assert fig.layout_updates[1] == {"height": 512}
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_id_columns_for_name_collisions_only_when_player_ids_disambiguate() -> None:
    with_collision = pd.DataFrame(
        {
            "player_id": ["one", "two", "three"],
            "name_full": ["Chris Smith", "Chris Smith", "Taylor Traded"],
        }
    )
    without_collision = pd.DataFrame(
        {
            "player_id": ["one", "two"],
            "name_full": ["Chris Smith", "Taylor Traded"],
        }
    )
    missing_id = pd.DataFrame({"name_full": ["Chris Smith", "Chris Smith"]})

    assert id_columns_for_name_collisions(with_collision) == ["player_id"]
    assert id_columns_for_name_collisions(without_collision) == []
    assert id_columns_for_name_collisions(missing_id) == []


def test_scale_payroll_converts_display_columns_without_mutating_input() -> None:
    raw = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "salary": [1_500_000],
            "surplus_value": [2_000_000],
            "dead_money_share": [0.25],
            "wins": [90],
        }
    )

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 100
    assert scaled.loc[0, "salary"] == 1.5
    assert scaled.loc[0, "surplus_value"] == 2
    assert scaled.loc[0, "dead_money_share"] == 25
    assert scaled.loc[0, "wins"] == 90
    assert raw.loc[0, "payroll"] == 100_000_000
