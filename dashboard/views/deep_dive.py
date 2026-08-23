"""Team Deep Dive — franchise dossier."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import load_player_season_metrics
from dashboard.helpers import (
    add_payroll_millions,
    format_money_millions,
    format_signed_int,
    format_war,
    scale_money_columns,
)
from dashboard.ui import (
    SCATTER_MARKER as _SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    player_column_config,
    salary_note as _salary_note,
    scale_payroll as _scale_payroll,
    season_picker as _season_picker,
    show_table as _show_table,
    team_column_config,
    team_select,
)
from src.baseball_analytics.dashboard_utils import player_id_columns_for_duplicate_names

# Bound by app.py after data load (also injected by unit tests).
metrics: pd.DataFrame | None = None
all_teams: list[str] = []
all_years: list[int] = []

_TEAM_COL_CFG = team_column_config()
_PLAYER_COL_CFG = player_column_config()


def page_team_deep_dive() -> None:
    _page_header("Team Deep Dive")
    if metrics is None:
        _empty("metrics")
        return
    if not all_teams:
        _empty("team")
        return

    c1, c2 = st.columns([3, 3])
    with c1:
        team = team_select(all_teams)
    with c2:
        year = _season_picker()

    team_history = metrics[metrics["team_name"] == team].sort_values("year_id")
    if team_history.empty:
        _empty("team")
        return

    season_row = team_history[team_history["year_id"] == year] if year is not None else team_history.iloc[0:0]
    phase = ""
    if not season_row.empty and "window_phase" in season_row.columns:
        phase = str(season_row.iloc[0].get("window_phase") or "")
    badge = f'<span class="phase-badge">{phase.title()}</span>' if phase and phase != "nan" else ""
    st.markdown(
        f'<div class="dossier-title"><h2>{team} — {year if year is not None else "—"}</h2>{badge}</div>',
        unsafe_allow_html=True,
    )
    if year is not None:
        _salary_note(year)

    if not season_row.empty:
        r = season_row.iloc[0]
        kpis = [
            ("Wins", int(r["wins"]) if pd.notna(r.get("wins")) else "—"),
            ("Losses", int(r["losses"]) if pd.notna(r.get("losses")) else "—"),
            ("Run Diff", format_signed_int(r.get("run_diff"))),
            ("Payroll", format_money_millions(r.get("payroll"))),
            ("Team WAR", format_war(r.get("team_total_war"))),
            ("Surplus", format_money_millions(r.get("surplus_value"))),
            ("$/WAR", format_money_millions(r.get("cost_per_war"), decimals=1)),
            ("Phase", str(r.get("window_phase", "—")).title()),
        ]
        cols = st.columns(len(kpis))
        for col, (label, value) in zip(cols, kpis):
            col.metric(label, value)
    else:
        st.info(f"No row for {team} in {year}. History for other seasons is below.")

    panel_head("Season history", "Newest first")
    hist_cols = [c for c in [
        "year_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "war_source", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ] if c in team_history.columns]
    _show_table(
        scale_money_columns(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True),
        _TEAM_COL_CFG,
        height=280,
    )

    panel_head("Trajectory", "Wins vs Pythagorean, payroll, and WAR")
    ch1, ch2, ch3 = st.columns(3)
    with ch1:
        fig_w = px.line(team_history, x="year_id", y="wins", markers=True, title="Wins")
        if "pythag_wins" in team_history.columns:
            fig_w.add_scatter(
                x=team_history["year_id"],
                y=team_history["pythag_wins"],
                mode="lines",
                name="Pythag W",
                line=dict(dash="dash", color="#64748b"),
            )
        fig_w.update_layout(xaxis_title="Season", yaxis_title="Wins")
        _chart(fig_w, height=280)
    with ch2:
        if team_history["payroll"].notna().any():
            pay = add_payroll_millions(team_history)
            fig_p = px.bar(pay, x="year_id", y="payroll_m", title="Payroll ($M)", color_discrete_sequence=["#e11d2e"])
            fig_p.update_layout(xaxis_title="Season", yaxis_title="Payroll ($M)")
            fig_p.update_traces(hovertemplate="Year: %{x}<br>Payroll: $%{y:.1f}M<extra></extra>")
            _chart(fig_p, height=280)
        else:
            st.caption("No payroll history for this franchise.")
    with ch3:
        if "team_total_war" in team_history.columns and team_history["team_total_war"].notna().any():
            fig_war = px.line(team_history, x="year_id", y="team_total_war", markers=True, title="Team WAR")
            fig_war.update_layout(xaxis_title="Season", yaxis_title="Team WAR")
            _chart(fig_war, height=280)
        elif "window_phase" in team_history.columns:
            fig_ph = px.scatter(
                team_history,
                x="year_id",
                y="window_phase",
                color="window_phase",
                title="Window phase",
            )
            fig_ph.update_layout(xaxis_title="Season", yaxis_title="Phase")
            fig_ph.update_traces(marker=_SCATTER_MARKER)
            _chart(fig_ph, height=280)

    panel_head(f"Roster — {year if year is not None else '—'}", "Sorted by WAR")
    players = load_player_season_metrics()
    if players is not None:
        roster = players[(players["year_id"] == year)]
        if "team_name" in roster.columns:
            roster = roster[roster["team_name"] == team]
        if not roster.empty:
            roster_id = player_id_columns_for_duplicate_names(roster)
            roster_cols = [c for c in (roster_id + [
                "name_full", "player_type", "pa", "hr", "bb", "woba", "batting_war",
                "ip", "era", "fip", "pitching_war",
                "player_war", "war_source", "salary", "surplus_value", "contract_label",
            ]) if c in roster.columns]
            _show_table(
                _scale_payroll(roster[roster_cols]).sort_values("player_war", ascending=False).reset_index(drop=True),
                _PLAYER_COL_CFG, height=500,
            )
        else:
            st.info("No player data for this team/season.")
