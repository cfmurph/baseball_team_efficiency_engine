from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.baseball_analytics.dashboard_utils import (
    compute_slider_max,
    player_id_columns_for_name_collisions,
    render_plotly_chart,
)


class _StreamlitRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[go.Figure, bool]] = []

    def plotly_chart(self, fig: go.Figure, use_container_width: bool) -> None:
        self.calls.append((fig, use_container_width))


def test_compute_slider_max_uses_current_year_when_no_metric_years() -> None:
    assert compute_slider_max([], current_year=2026) == 2026


def test_compute_slider_max_keeps_future_metric_year_available() -> None:
    assert compute_slider_max([2020, 2028], current_year=2026) == 2028


def test_render_plotly_chart_applies_layout_and_delegates_to_streamlit() -> None:
    fig = go.Figure()
    recorder = _StreamlitRecorder()

    render_plotly_chart(
        fig,
        recorder,
        {"paper_bgcolor": "#0d1117", "plot_bgcolor": "#0d1117"},
        height=512,
    )

    assert fig.layout.paper_bgcolor == "#0d1117"
    assert fig.layout.plot_bgcolor == "#0d1117"
    assert fig.layout.height == 512
    assert recorder.calls == [(fig, True)]


def test_player_id_columns_for_name_collisions_only_when_needed() -> None:
    duplicated_names = pd.DataFrame(
        {
            "player_id": ["smith-j-1", "smith-j-2", "unique"],
            "name_full": ["Jordan Smith", "Jordan Smith", "Unique Player"],
        }
    )
    unique_names = pd.DataFrame(
        {
            "player_id": ["smith-j-1", "unique"],
            "name_full": ["Jordan Smith", "Unique Player"],
        }
    )
    no_id = pd.DataFrame({"name_full": ["Jordan Smith", "Jordan Smith"]})

    assert player_id_columns_for_name_collisions(duplicated_names) == ["player_id"]
    assert player_id_columns_for_name_collisions(unique_names) == []
    assert player_id_columns_for_name_collisions(no_id) == []
