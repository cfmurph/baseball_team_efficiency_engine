"""Roster Lab — player WAR vs salary."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import load_player_season_metrics, load_sr_player_metrics
from dashboard.helpers import (
    CONTRACT_COLORS,
    scale_money_columns,
    teams_from_frame,
    years_from_frame,
)
from dashboard.state import SEASON_YEAR, SELECTED_TEAM
from dashboard.ui import (
    SCATTER_MARKER as _SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    player_column_config,
    salary_note as _salary_note,
    show_table as _show_table,
)
from src.baseball_analytics.dashboard_utils import player_id_columns_for_duplicate_names

_PLAYER_COL_CFG = player_column_config()


def page_player_explorer() -> None:
    _page_header(
        "Roster Lab",
        extra_caption=(
            "WAR is Baseball-Reference rWAR when the player-season maps "
            "(war_source=real); otherwise the Lahman approximation."
        ),
    )

    players = load_player_season_metrics()
    sr_players = load_sr_player_metrics()
    if players is None:
        _empty("players")
        return

    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
    with f1:
        yr_opts = years_from_frame(players)
        if yr_opts and st.session_state.get(SEASON_YEAR) not in yr_opts:
            st.session_state[SEASON_YEAR] = yr_opts[-1]
        year = st.selectbox("Season", yr_opts, key=SEASON_YEAR) if yr_opts else None
    with f2:
        team_opts = ["All Teams"] + teams_from_frame(players)
        if "roster_team_filter" not in st.session_state:
            current = st.session_state.get(SELECTED_TEAM)
            st.session_state["roster_team_filter"] = current if current in team_opts else "All Teams"
        team = st.selectbox("Team", team_opts, key="roster_team_filter")
        if team != "All Teams":
            st.session_state[SELECTED_TEAM] = team
    with f3:
        type_opts = ["All Types"]
        if "player_type" in players.columns:
            type_opts += sorted(players["player_type"].dropna().unique().tolist())
        ptype = st.selectbox("Type", type_opts, key="pe_type")
    with f4:
        name_search = st.text_input("Search player name", key="pe_name", placeholder="e.g. Judge")
    with f5:
        sort_col_opts = ["player_war", "salary", "surplus_value", "batting_war", "pitching_war", "pa", "hr", "ip", "era", "fip", "woba", "war_source"]
        sort_col_opts = [c for c in sort_col_opts if c in players.columns]
        sort_by = st.selectbox("Sort by", sort_col_opts, key="pe_sort")

    _salary_note(year)
    filt = players[players["year_id"] == year].copy() if year is not None else players.copy()
    if team != "All Teams" and "team_name" in filt.columns:
        filt = filt[filt["team_name"] == team]
    if ptype != "All Types" and "player_type" in filt.columns:
        filt = filt[filt["player_type"] == ptype]
    if name_search and "name_full" in filt.columns:
        filt = filt[filt["name_full"].str.contains(name_search, case=False, na=False)]
    if sort_col_opts:
        filt = filt.sort_values(sort_by, ascending=(sort_by in ["era", "fip"]), na_position="last").reset_index(drop=True)

    st.caption(f"{len(filt):,} players shown")
    if filt.empty:
        _empty("generic")
        return

    if "salary" in filt.columns and "player_war" in filt.columns:
        plot_f = scale_money_columns(filt.dropna(subset=["salary", "player_war"]))
        if not plot_f.empty:
            panel_head("WAR vs salary", "Color = contract classification")
            fig = px.scatter(
                plot_f,
                x="salary",
                y="player_war",
                color="contract_label" if "contract_label" in plot_f.columns else None,
                hover_name="name_full" if "name_full" in plot_f.columns else None,
                hover_data=[c for c in ["team_name", "year_id"] if c in plot_f.columns],
                labels={"salary": "Salary ($M)", "player_war": "WAR", "contract_label": "Contract"},
                color_discrete_map=CONTRACT_COLORS,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#1e2836")
            fig.update_traces(marker=_SCATTER_MARKER)
            _chart(fig, height=420)

    tab_bat, tab_pit, tab_contract, tab_all = st.tabs(["Batting", "Pitching", "Contract", "All Stats"])

    has_name_collision = (
        "name_full" in filt.columns
        and filt.duplicated("name_full", keep=False).any()
    )
    id_col = player_id_columns_for_duplicate_names(filt)
    if has_name_collision:
        st.caption("Multiple players share a name in this view — the Player ID column distinguishes them.")

    bat_cols = id_col + ["name_full", "team_name", "player_type", "pa", "hr", "bb", "woba", "batting_war", "war_source"]
    pit_cols = id_col + ["name_full", "team_name", "player_type", "ip", "era", "fip", "pitching_war", "war_source"]
    contract_cols = id_col + ["name_full", "team_name", "player_type", "player_war", "war_source", "salary", "surplus_value", "contract_label"]
    all_cols = [c for c in (id_col + [
        "name_full", "team_name", "player_type",
        "pa", "hr", "bb", "woba", "batting_war",
        "ip", "era", "fip", "pitching_war",
        "player_war", "war_source", "salary", "surplus_value", "contract_label",
    ]) if c in filt.columns]

    with tab_bat:
        _show_table(scale_money_columns(filt[[c for c in bat_cols if c in filt.columns]]), _PLAYER_COL_CFG)
    with tab_pit:
        pit = filt[filt["ip"].notna() & (filt["ip"] > 0)] if "ip" in filt.columns else filt
        _show_table(scale_money_columns(pit[[c for c in pit_cols if c in pit.columns]]), _PLAYER_COL_CFG)
    with tab_contract:
        _show_table(scale_money_columns(filt[[c for c in contract_cols if c in filt.columns]]), _PLAYER_COL_CFG)
    with tab_all:
        _show_table(scale_money_columns(filt[all_cols]), _PLAYER_COL_CFG)

    if sr_players is not None and not sr_players.empty:
        st.divider()
        panel_head("Sportradar", "Real WAR · wRC+ · ERA-")
        sr_yr_opts = years_from_frame(sr_players)
        sr_year = st.selectbox("SR Season", sr_yr_opts, index=len(sr_yr_opts) - 1, key="pe_sr_year")
        sr_filt = sr_players[sr_players["year_id"] == sr_year].copy()
        if team != "All Teams" and "team_name" in sr_filt.columns:
            sr_filt = sr_filt[sr_filt["team_name"] == team]
        name_col = "full_name" if "full_name" in sr_filt.columns else None
        if name_search and name_col:
            sr_filt = sr_filt[sr_filt[name_col].str.contains(name_search, case=False, na=False)]
        if "player_war_sr" in sr_filt.columns:
            sr_filt = sr_filt.sort_values("player_war_sr", ascending=False, na_position="last").reset_index(drop=True)
        sr_display_cols = [c for c in ["full_name", "team_id", "primary_position", "pa", "hr", "woba", "wrc_plus", "war", "bwar", "fwar", "ip", "era", "era_minus", "fip", "k9", "p_war", "player_war_sr"] if c in sr_filt.columns]
        sr_col_cfg = {
            "full_name": st.column_config.TextColumn("Player", width="medium"),
            "team_id": st.column_config.TextColumn("Team"),
            "primary_position": st.column_config.TextColumn("Pos", width="small"),
            "pa": st.column_config.NumberColumn("PA", format="%d"),
            "hr": st.column_config.NumberColumn("HR", format="%d"),
            "woba": st.column_config.NumberColumn("wOBA", format="%.3f"),
            "wrc_plus": st.column_config.NumberColumn("wRC+", format="%.0f"),
            "war": st.column_config.NumberColumn("WAR (bat)", format="%.1f"),
            "bwar": st.column_config.NumberColumn("bWAR", format="%.1f"),
            "fwar": st.column_config.NumberColumn("fWAR", format="%.1f"),
            "ip": st.column_config.NumberColumn("IP", format="%.1f"),
            "era": st.column_config.NumberColumn("ERA", format="%.2f"),
            "era_minus": st.column_config.NumberColumn("ERA-", format="%.1f"),
            "fip": st.column_config.NumberColumn("FIP", format="%.2f"),
            "k9": st.column_config.NumberColumn("K/9", format="%.1f"),
            "p_war": st.column_config.NumberColumn("pWAR", format="%.1f"),
            "player_war_sr": st.column_config.NumberColumn("Total WAR", format="%.1f"),
        }
        st.caption(f"{len(sr_filt):,} players")
        _show_table(sr_filt[sr_display_cols], sr_col_cfg)
