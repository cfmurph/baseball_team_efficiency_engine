"""Overview — league command center."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.data import load_window_phases
from dashboard.helpers import (
    add_payroll_millions,
    apply_efficiency_labels,
    filter_season,
    kpi_cards_html,
    leaderboard_html,
    overview_kpi_payload,
    rank_by_efficiency,
    scale_money_columns,
    top_n_by,
)
from dashboard.ui import (
    SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    league_select,
    salary_note as _salary_note,
    season_picker as _season_picker_impl,
    show_table as _show_table,
    team_column_config,
)

_TEAM_COL_CFG = None


def _cfg() -> dict:
    global _TEAM_COL_CFG
    if _TEAM_COL_CFG is None:
        _TEAM_COL_CFG = team_column_config()
    return _TEAM_COL_CFG


metrics: pd.DataFrame | None = None
all_years: list[int] = []


def page_league_snapshot() -> None:
    _page_header(
        "Overview",
        extra_caption=(
            "Team WAR is Baseball-Reference rWAR rolled up from players "
            "(war_source=real); Lahman wOBA/FIP approx is the fallback."
        ),
    )
    if metrics is None:
        _empty("metrics")
        return

    with st.container():
        col_nav, col_lg = st.columns([3, 1])
        with col_nav:
            year = _season_picker_impl()
        with col_lg:
            lg = league_select()
    if year is None:
        _empty("season")
        return

    season = apply_efficiency_labels(filter_season(metrics, year, lg))
    _salary_note(year)

    if season.empty:
        _empty("season")
        return

    cards = overview_kpi_payload(season)
    st.markdown(kpi_cards_html(cards), unsafe_allow_html=True)

    ranked = rank_by_efficiency(season)
    extra = tuple(c for c in ("wins", "payroll", "team_total_war", "wins_per_10m") if c in season.columns)
    cheap = top_n_by(season, "surplus_value", n=5, extra_cols=extra)
    dear = top_n_by(season, "surplus_value", n=5, ascending=True, extra_cols=extra)

    if not cheap.empty:
        left, right = st.columns(2)
        with left:
            panel_head("Buying wins cheaply", "Highest surplus value vs market payroll")
            cheap_m = scale_money_columns(cheap)
            if "surplus_value" in cheap_m.columns:
                st.markdown(
                    leaderboard_html(cheap_m, value_col="surplus_value", value_format="{:+.0f}", suffix="M"),
                    unsafe_allow_html=True,
                )
            else:
                _show_table(cheap_m, _cfg(), height=220)
        with right:
            panel_head("Paying above market", "Lowest surplus — expensive relative to WAR")
            dear_m = scale_money_columns(dear)
            if "surplus_value" in dear_m.columns:
                st.markdown(
                    leaderboard_html(dear_m, value_col="surplus_value", value_format="{:+.0f}", suffix="M"),
                    unsafe_allow_html=True,
                )
            else:
                _show_table(dear_m, _cfg(), height=220)

    plot_df = add_payroll_millions(season.dropna(subset=["payroll", "wins"]))
    chart_col, rank_col = st.columns([7, 5])
    with chart_col:
        panel_head(f"{year} payroll vs wins", "Color = surplus value")
        if not plot_df.empty:
            if "surplus_value" in plot_df.columns:
                plot_df["surplus_m"] = plot_df["surplus_value"] / 1_000_000
            if "cost_per_war" in plot_df.columns:
                plot_df["cost_per_war_m"] = plot_df["cost_per_war"] / 1_000_000
            color_col = "surplus_m" if "surplus_m" in plot_df.columns and plot_df["surplus_m"].notna().any() else (
                "window_phase" if "window_phase" in plot_df.columns else "league_id"
            )
            hover = [c for c in ["team_total_war", "wins_per_10m", "surplus_m", "cost_per_war_m"] if c in plot_df.columns]
            fig = px.scatter(
                plot_df,
                x="payroll_m",
                y="wins",
                color=color_col,
                hover_name="team_name",
                hover_data=hover,
                labels={
                    "payroll_m": "Payroll ($M)",
                    "wins": "Wins",
                    "surplus_m": "Surplus ($M)",
                    "team_total_war": "Team WAR",
                    "wins_per_10m": "W/$10M",
                    "cost_per_war_m": "$/WAR ($M)",
                },
                color_continuous_scale=["#e11d2e", "#f59e0b", "#22c55e"] if color_col == "surplus_m" else None,
            )
            fig.update_traces(marker=SCATTER_MARKER)
            fig.update_layout(coloraxis_colorbar_title="Surplus ($M)" if color_col == "surplus_m" else None)
            _chart(fig, height=440)
        else:
            st.caption("No payroll values to plot for this season.")
    with rank_col:
        panel_head("Efficiency ranking", "Surplus, then W/$10M")
        compact_cols = [c for c in ["rank", "team_name", "wins", "surplus_value", "cost_per_war", "team_total_war", "window_phase"] if c in ranked.columns]
        _show_table(scale_money_columns(ranked[compact_cols]), _cfg(), height=440)

    rank_tab, standings_tab, phase_tab = st.tabs(["Full ranking", "Standings", "Window phases"])

    table_cols = [
        "rank", "team_name", "league_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "war_source", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ]
    with rank_tab:
        st.caption("Sorted by surplus value, then wins per $10M. Click a header to re-sort.")
        display_cols = [c for c in table_cols if c in ranked.columns]
        _show_table(scale_money_columns(ranked[display_cols]), _cfg(), height=560)

    with standings_tab:
        leagues = [value for value in ("AL", "NL") if "league_id" in season.columns and (season["league_id"] == value).any()]
        stand_priority = ["team_name", "wins", "losses", "run_diff", "team_total_war", "payroll", "wins_per_10m", "window_phase"]
        if not leagues:
            _show_table(
                scale_money_columns(ranked[[c for c in ["rank", "team_name", "wins", "losses", "run_diff", "window_phase"] if c in ranked.columns]]),
                _cfg(),
                height=420,
            )
        else:
            cols = st.columns(len(leagues))
            stand_cols = [c for c in stand_priority if c in season.columns]
            for col, lg_name in zip(cols, leagues):
                with col:
                    st.subheader(lg_name)
                    lg_df = season[season["league_id"] == lg_name].sort_values("wins", ascending=False)
                    _show_table(scale_money_columns(lg_df[stand_cols]).reset_index(drop=True), _cfg(), height=360)

    with phase_tab:
        window_df = load_window_phases()
        if window_df is None:
            if "window_phase" not in season.columns:
                _empty("window")
            else:
                phases = ["All"] + sorted(season["window_phase"].dropna().astype(str).unique().tolist())
                phase_filter = st.selectbox("Filter by phase", phases, key="ov_phase_season")
                phase_view = season if phase_filter == "All" else season[season["window_phase"] == phase_filter]
                cols = [c for c in ["team_name", "wins", "payroll", "team_total_war", "window_phase"] if c in phase_view.columns]
                _show_table(scale_money_columns(phase_view[cols]).sort_values("wins", ascending=False).reset_index(drop=True), _cfg(), height=420)
        else:
            display = window_df.copy()
            if "payroll" in display.columns:
                display["payroll"] = display["payroll"] / 1_000_000
            phases = ["All"] + sorted(display["window_phase"].dropna().astype(str).unique().tolist()) if "window_phase" in display.columns else ["All"]
            phase_filter = st.selectbox("Filter by phase", phases, key="ov_phase")
            if phase_filter != "All":
                display = display[display["window_phase"] == phase_filter]
            phase_cfg = {
                **_cfg(),
                "year_id": st.column_config.NumberColumn("Latest Year", format="%d"),
            }
            _show_table(display.sort_values("wins", ascending=False).reset_index(drop=True), phase_cfg, height=480)
