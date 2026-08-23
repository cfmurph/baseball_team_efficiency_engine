"""Contract Watch — surplus, overpays, dead money."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import load_player_season_metrics
from dashboard.helpers import CONTRACT_COLORS, scale_money_columns, teams_from_frame, years_from_frame
from dashboard.state import SEASON_YEAR, SELECTED_TEAM
from dashboard.ui import (
    SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    player_column_config,
    salary_note as _salary_note,
    show_table as _show_table,
)

_PLAYER_COL_CFG: dict | None = None


def _cfg() -> dict:
    global _PLAYER_COL_CFG
    if _PLAYER_COL_CFG is None:
        _PLAYER_COL_CFG = player_column_config()
    return _PLAYER_COL_CFG


def page_contract_analysis() -> None:
    _page_header(
        "Contract Watch",
        extra_caption=(
            "Every player contract, classified and searchable. Salary data from Lahman (through 2016). "
            "Surplus value uses Baseball-Reference rWAR when war_source=real."
        ),
    )

    players = load_player_season_metrics()
    if players is None:
        _empty("players")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        yr_opts = years_from_frame(players)
        season_opts = ["All Seasons"] + yr_opts
        if "contracts_season_filter" not in st.session_state:
            current = st.session_state.get(SEASON_YEAR)
            st.session_state["contracts_season_filter"] = current if current in yr_opts else "All Seasons"
        year = st.selectbox("Season", season_opts, key="contracts_season_filter")
        if year != "All Seasons":
            st.session_state[SEASON_YEAR] = int(year)
    with f2:
        team_opts = ["All Teams"] + teams_from_frame(players)
        if "contracts_team_filter" not in st.session_state:
            current_team = st.session_state.get(SELECTED_TEAM)
            st.session_state["contracts_team_filter"] = current_team if current_team in team_opts else "All Teams"
        team = st.selectbox("Team", team_opts, key="contracts_team_filter")
        if team != "All Teams":
            st.session_state[SELECTED_TEAM] = team
    with f3:
        name_search = st.text_input("Search player", key="ca_name", placeholder="e.g. Bonds")

    if year != "All Seasons":
        _salary_note(int(year))

    filt = players.copy()
    if year != "All Seasons":
        filt = filt[filt["year_id"] == int(year)]
    if team != "All Teams" and "team_name" in filt.columns:
        filt = filt[filt["team_name"] == team]
    if name_search and "name_full" in filt.columns:
        filt = filt[filt["name_full"].str.contains(name_search, case=False, na=False)]
    if "salary" in filt.columns:
        filt = filt[filt["salary"] > 0]
    if filt.empty:
        _empty("generic")
        return

    contract_cols = [c for c in [
        "name_full", "year_id", "team_name", "player_type",
        "player_war", "war_source", "salary", "surplus_value", "contract_label",
        "batting_war", "pitching_war", "pa", "ip",
    ] if c in filt.columns]
    tabs = st.tabs(["All Contracts", "Surplus Value", "Overpaid", "Dead Money", "Fair Value"])

    def _contract_table(df: pd.DataFrame, sort: str, asc: bool = False) -> None:
        if df.empty:
            _empty("generic")
            return
        display = scale_money_columns(df[contract_cols]).sort_values(sort, ascending=asc, na_position="last").reset_index(drop=True)
        st.caption(f"{len(display):,} contracts")
        _show_table(display, _cfg())

    labels = filt["contract_label"] if "contract_label" in filt.columns else pd.Series(dtype=str)
    with tabs[0]:
        _contract_table(filt, "surplus_value", asc=False)
    with tabs[1]:
        sv = filt[labels == "surplus_value"] if "contract_label" in filt.columns else filt[filt["surplus_value"] > 2e6]
        _contract_table(sv, "surplus_value", asc=False)
    with tabs[2]:
        op = filt[labels == "overpaid"] if "contract_label" in filt.columns else filt
        _contract_table(op, "surplus_value", asc=True)
    with tabs[3]:
        dm = filt[labels == "dead_money"] if "contract_label" in filt.columns else filt
        _contract_table(dm, "salary", asc=False)
    with tabs[4]:
        fv = filt[labels == "fair_value"] if "contract_label" in filt.columns else filt
        _contract_table(fv, "player_war", asc=False)

    if "salary" in filt.columns and "player_war" in filt.columns:
        plot_f = scale_money_columns(filt.dropna(subset=["salary", "player_war"]))
        panel_head("WAR vs salary")
        fig = px.scatter(
            plot_f,
            x="salary",
            y="player_war",
            color="contract_label" if "contract_label" in plot_f.columns else None,
            hover_name="name_full" if "name_full" in plot_f.columns else None,
            hover_data=[c for c in ["year_id", "team_name"] if c in plot_f.columns],
            labels={"salary": "Salary ($M)", "player_war": "WAR", "contract_label": "Contract"},
            color_discrete_map=CONTRACT_COLORS,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#243044")
        fig.update_traces(marker=SCATTER_MARKER)
        _chart(fig, height=450)
