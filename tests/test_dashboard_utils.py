from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    has_duplicate_names,
    player_id_columns_for_collisions,
    render_plotly_chart,
    scale_payroll_for_display,
    slider_max_year,
)


def test_scale_payroll_for_display_converts_money_and_shares_without_mutating_input() -> None:
    source = pd.DataFrame(
        {
            "payroll": [120_000_000],
            "salary": [7_500_000],
            "surplus_value": [-2_000_000],
            "dead_money_share": [0.125],
            "wins": [90],
        }
    )

    result = scale_payroll_for_display(source)

    assert result.loc[0, "payroll"] == 120
    assert result.loc[0, "salary"] == 7.5
    assert result.loc[0, "surplus_value"] == -2
    assert result.loc[0, "dead_money_share"] == 12.5
    assert result.loc[0, "wins"] == 90
    assert source.loc[0, "payroll"] == 120_000_000
    assert source.loc[0, "dead_money_share"] == 0.125


def test_slider_max_year_uses_current_year_when_metrics_have_no_years() -> None:
    assert slider_max_year([], current_year=2026) == 2026
    assert slider_max_year([None, pd.NA], current_year=2026) == 2026


def test_slider_max_year_keeps_future_artifact_year_selectable() -> None:
    assert slider_max_year([1990, 2024, 2028], current_year=2026) == 2028
    assert slider_max_year([1990, 2024], current_year=2026) == 2026


class FakeFigure:
    def __init__(self) -> None:
        self.layout_updates: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layout_updates.append(kwargs)


class FakeStreamlit:
    def __init__(self) -> None:
        self.plotly_chart_calls: list[tuple[FakeFigure, bool]] = []

    def plotly_chart(self, fig: FakeFigure, use_container_width: bool) -> None:
        self.plotly_chart_calls.append((fig, use_container_width))


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = FakeFigure()
    streamlit = FakeStreamlit()

    render_plotly_chart(streamlit, fig, height=320)

    assert fig.layout_updates[0]["template"] == "plotly_dark"
    assert fig.layout_updates[0]["paper_bgcolor"] == "#0d1117"
    assert fig.layout_updates[1] == {"height": 320}
    assert streamlit.plotly_chart_calls == [(fig, True)]


def test_player_id_columns_are_shown_only_for_same_name_collisions() -> None:
    players = pd.DataFrame(
        {
            "name_full": ["Will Smith", "Will Smith", "Mookie Betts"],
            "player_id": ["smithwi01", "smithwi02", "bettsmo01"],
        }
    )

    assert has_duplicate_names(players)
    assert player_id_columns_for_collisions(players) == ["player_id"]
    assert player_id_columns_for_collisions(players.drop(columns=["player_id"])) == []
    assert player_id_columns_for_collisions(players.iloc[[0, 2]]) == []
