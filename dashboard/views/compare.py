"""Compare Teams — multi-franchise trends."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.helpers import metric_label, scale_money_columns
from dashboard.state import SELECTED_TEAM
from dashboard.ui import (
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    salary_note as _salary_note,
    show_table as _show_table,
    team_column_config,
)

metrics: pd.DataFrame | None = None
all_teams: list[str] = []
_slider_lo: int = 1990
_slider_max: int = 2016

_TEAM_COL_CFG: dict | None = None


def _cfg() -> dict:
    global _TEAM_COL_CFG
    if _TEAM_COL_CFG is None:
        _TEAM_COL_CFG = team_column_config()
    return _TEAM_COL_CFG


def page_compare_teams() -> None:
    _page_header("Compare Teams")
    if metrics is None:
        _empty("metrics")
        return
    if len(all_teams) < 2:
        _empty("compare")
        return

    c1, c2 = st.columns([4, 2])
    with c1:
        seed = st.session_state.get(SELECTED_TEAM)
        default_teams = [seed] + [t for t in all_teams[:4] if t != seed] if seed in all_teams else all_teams[:4]
        selected = st.multiselect("Teams", all_teams, default=default_teams[:4], key="compare_teams")
        if len(selected) == 1:
            st.session_state[SELECTED_TEAM] = selected[0]
    with c2:
        year_range = st.slider(
            "Years",
            int(_slider_lo),
            int(_slider_max),
            (max(int(_slider_lo), int(_slider_max) - 9), int(_slider_max)),
            key="sc_range",
        )

    if len(selected) < 2:
        _empty("compare")
        return

    compare_df = metrics[
        metrics["team_name"].isin(selected) & metrics["year_id"].between(year_range[0], year_range[1])
    ].copy()
    if compare_df.empty:
        _empty("generic")
        return

    latest_year = int(compare_df["year_id"].max())
    _salary_note(latest_year)
    panel_head(f"Latest season in range — {latest_year}", "Sorted by wins")
    latest = compare_df[compare_df["year_id"] == latest_year]
    table_cols = [c for c in [
        "team_name", "wins", "losses", "run_diff", "pythag_wins",
        "payroll", "wins_per_10m", "team_total_war", "cost_per_war",
        "surplus_value", "gini_salary", "window_phase",
    ] if c in latest.columns]
    _show_table(scale_money_columns(latest[table_cols]).sort_values("wins", ascending=False).reset_index(drop=True), _cfg(), height=250)

    panel_head(f"History — {year_range[0]}–{year_range[1]}")
    hist_cols = [c for c in [
        "year_id", "team_name", "wins", "run_diff", "payroll",
        "wins_per_10m", "team_total_war", "surplus_value", "window_phase",
    ] if c in compare_df.columns]
    _show_table(
        scale_money_columns(compare_df[hist_cols]).sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True),
        _cfg(),
        height=400,
    )

    metric_opts = [c for c in ["wins", "payroll", "wins_per_10m", "team_total_war", "cost_per_war", "surplus_value", "run_diff", "gini_salary"] if c in compare_df.columns]
    y_metric = st.selectbox("Chart metric", metric_opts, format_func=metric_label, key="sc_metric")
    plot_df = compare_df.copy()
    if y_metric in {"payroll", "cost_per_war", "surplus_value"}:
        plot_df[y_metric] = plot_df[y_metric] / 1_000_000
    y_label = metric_label(y_metric)
    fig = px.line(
        plot_df,
        x="year_id",
        y=y_metric,
        color="team_name",
        markers=True,
        title=f"{y_label} — {year_range[0]}–{year_range[1]}",
        labels={y_metric: y_label, "year_id": "Season", "team_name": "Team"},
    )
    _chart(fig, height=380)
