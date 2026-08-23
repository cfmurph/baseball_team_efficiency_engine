"""What-If Sim — payroll impact projection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.helpers import format_money_millions, format_ratio, format_war, scale_money_columns, scoreboard_html
from dashboard.theme import AMBER
from dashboard.ui import (
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    salary_note as _salary_note,
    season_picker as _season_picker,
    show_table as _show_table,
    team_column_config,
    team_select,
)

metrics: pd.DataFrame | None = None
all_teams: list[str] = []
all_years: list[int] = []

_TEAM_COL_CFG = team_column_config()


def page_whatif() -> None:
    _page_header("What-If Sim")
    if metrics is None:
        _empty("metrics")
        return
    if not all_teams:
        _empty("team")
        return

    c1, c2 = st.columns(2)
    with c1:
        team = team_select(all_teams)
    with c2:
        year = _season_picker()

    team_history = metrics[metrics["team_name"] == team].sort_values("year_id")
    if team_history.empty:
        _empty("team")
        return

    row = team_history[team_history["year_id"] == year] if year is not None else team_history.iloc[0:0]
    if row.empty:
        row = team_history.iloc[[-1]]
        st.info(f"No data for {year} — showing {int(row.iloc[0]['year_id'])}")
    r = row.iloc[0]
    current_payroll = float(r.get("payroll", 0) or 0)
    current_wins = float(r.get("wins", 0) or 0)
    current_war = r.get("team_total_war")
    _salary_note(int(r["year_id"]))

    panel_head(f"{team} — {int(r['year_id'])} baseline")
    st.markdown(
        scoreboard_html([
            ("Payroll", format_money_millions(current_payroll)),
            ("Wins", int(current_wins)),
            ("W/$10M", format_ratio(r.get("wins_per_10m"))),
            ("Team WAR", format_war(current_war)),
        ]),
        unsafe_allow_html=True,
    )

    st.divider()
    payroll_delta_m = st.slider("Payroll change ($M)", -50, 150, 20, step=5, key="whatif_payroll_delta")
    new_payroll = current_payroll + payroll_delta_m * 1_000_000
    valid = metrics.dropna(subset=["payroll", "wins"])
    if len(valid) > 10:
        coeffs = np.polyfit(valid["payroll"].values, valid["wins"].values, 1)
        win_gain = coeffs[0] * (payroll_delta_m * 1_000_000)
    else:
        win_gain = 0.0
    projected_wins = current_wins + win_gain

    st.markdown(
        scoreboard_html([
            ("New payroll", f"{format_money_millions(new_payroll)}  ({payroll_delta_m:+.0f}M)"),
            ("Projected wins", f"{projected_wins:.0f}  ({win_gain:+.1f})"),
            ("New $/win", format_money_millions(new_payroll / max(projected_wins, 1), decimals=2) if projected_wins > 0 else "—"),
        ]),
        unsafe_allow_html=True,
    )
    st.caption("Linear regression on all historical team-seasons. Actual results depend on how the extra payroll is allocated.")

    panel_head("Historical record")
    hist_cols = [c for c in ["year_id", "wins", "run_diff", "payroll", "wins_per_10m", "team_total_war", "window_phase"] if c in team_history.columns]
    _show_table(
        scale_money_columns(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True),
        _TEAM_COL_CFG,
        height=350,
    )
    panel_head("Win history", "Star = projected season")
    fig = px.line(team_history, x="year_id", y="wins", markers=True)
    fig.add_scatter(
        x=[int(r["year_id"]) + 1],
        y=[projected_wins],
        mode="markers+text",
        marker=dict(color=AMBER, size=14, symbol="star"),
        text=["Projected"],
        textposition="top center",
        name="Projection",
    )
    fig.update_layout(xaxis_title="Season", yaxis_title="Wins")
    _chart(fig, height=320)
