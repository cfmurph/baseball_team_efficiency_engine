from __future__ import annotations

import pandas as pd

from src.baseball_analytics.dashboard_utils import (
    calculate_slider_max,
    id_column_for_name_collisions,
    render_plotly_chart,
    scale_payroll,
)


def test_scale_payroll_converts_display_units_without_mutating_source() -> None:
    source = pd.DataFrame(
        {
            "team_name": ["Athletics"],
            "payroll": [125_000_000],
            "salary": [7_500_000],
            "surplus_value": [-2_000_000],
            "dead_money_share": [0.125],
        }
    )

    result = scale_payroll(source)

    assert result.loc[0, "payroll"] == 125.0
    assert result.loc[0, "salary"] == 7.5
    assert result.loc[0, "surplus_value"] == -2.0
    assert result.loc[0, "dead_money_share"] == 12.5
    assert source.loc[0, "payroll"] == 125_000_000


def test_calculate_slider_max_uses_current_year_when_metrics_are_empty() -> None:
    assert calculate_slider_max([], current_year=2026) == 2026


def test_calculate_slider_max_allows_future_artifact_years() -> None:
    assert calculate_slider_max([2020, 2024, 2028], current_year=2026) == 2028


def test_id_column_for_name_collisions_only_when_player_ids_disambiguate() -> None:
    players = pd.DataFrame(
        {
            "player_id": ["smith001", "smith002", "jones001"],
            "name_full": ["Chris Smith", "Chris Smith", "Sam Jones"],
        }
    )

    assert id_column_for_name_collisions(players) == ["player_id"]
    assert id_column_for_name_collisions(players.drop(columns=["player_id"])) == []
    assert id_column_for_name_collisions(players.drop_duplicates("name_full")) == []


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    class FakeFigure:
        def __init__(self) -> None:
            self.layout_updates: list[dict] = []

        def update_layout(self, **kwargs) -> None:
            self.layout_updates.append(kwargs)

    rendered: list[tuple[FakeFigure, bool]] = []

    def fake_plotly_chart(fig: FakeFigure, *, use_container_width: bool) -> str:
        rendered.append((fig, use_container_width))
        return "rendered"

    fig = FakeFigure()
    result = render_plotly_chart(
        fig,
        fake_plotly_chart,
        {"template": "plotly_dark"},
        height=460,
    )

    assert result == "rendered"
    assert fig.layout_updates == [{"template": "plotly_dark"}, {"height": 460}]
    assert rendered == [(fig, True)]
