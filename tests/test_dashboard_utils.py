from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    id_columns_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
    season_range_defaults,
)


class DummyFigure:
    def __init__(self) -> None:
        self.layouts: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layouts.append(kwargs)


def test_compute_slider_max_handles_empty_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_uses_later_of_data_and_current_year() -> None:
    assert compute_slider_max([2019, 2024], current_year=2026) == 2026
    assert compute_slider_max([2019, 2028], current_year=2026) == 2028


def test_season_range_defaults_handles_empty_years() -> None:
    assert season_range_defaults([], current_year=2026) == (2026, 2026, (2026, 2026))


def test_season_range_defaults_uses_first_year_and_trailing_window() -> None:
    assert season_range_defaults([2001, 2005, 2020], current_year=2024) == (2001, 2024, (2015, 2024))


def test_scale_payroll_converts_display_units_without_mutating_input() -> None:
    raw = pd.DataFrame({
        "payroll": [200_000_000],
        "salary": [25_000_000],
        "surplus_value": [-5_000_000],
        "dead_money_share": [0.125],
    })

    scaled = scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 200
    assert scaled.loc[0, "salary"] == 25
    assert scaled.loc[0, "surplus_value"] == -5
    assert scaled.loc[0, "dead_money_share"] == pytest.approx(12.5)
    assert raw.loc[0, "payroll"] == 200_000_000


def test_id_columns_for_name_collisions_only_when_current_view_has_collision() -> None:
    same_name = pd.DataFrame({
        "player_id": ["smithjo01", "smithjo02"],
        "name_full": ["John Smith", "John Smith"],
    })
    unique_names = pd.DataFrame({
        "player_id": ["judgeaa01", "colege01"],
        "name_full": ["Aaron Judge", "Gerrit Cole"],
    })

    assert id_columns_for_name_collisions(same_name) == ["player_id"]
    assert id_columns_for_name_collisions(unique_names) == []
    assert id_columns_for_name_collisions(same_name.drop(columns=["player_id"])) == []


def test_render_plotly_chart_delegates_to_streamlit_once() -> None:
    fig = DummyFigure()
    streamlit = Mock()
    layout = {"template": "plotly_dark", "paper_bgcolor": "#0d1117"}

    render_plotly_chart(fig, streamlit, layout, height=460)

    assert fig.layouts == [layout, {"height": 460}]
    streamlit.plotly_chart.assert_called_once_with(fig, use_container_width=True)
