"""Efficiency Frontier — payroll-wins envelope + cluster archetypes."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import load_cluster_summary, load_frontier_data, load_team_clusters
from dashboard.helpers import add_payroll_millions, scale_money_columns, year_span_from_frame
from dashboard.theme import CRIMSON, CYAN, GREEN
from dashboard.state import SEASON_YEAR
from dashboard.ui import (
    SCATTER_MARKER as _SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    show_table as _show_table,
    team_column_config,
)

_TEAM_COL_CFG = team_column_config()


def page_efficiency_frontier() -> None:
    _page_header("Efficiency Frontier")
    frontier_data = load_frontier_data()
    clusters = load_team_clusters()
    frontier_tab, cluster_tab = st.tabs(["Frontier", "Team Archetypes"])

    with frontier_tab:
        if frontier_data is None:
            _empty("frontier")
        else:
            fd = frontier_data.copy()
            fd["above_label"] = fd["above_frontier"].map({True: "Above (efficient)", False: "Below (wasteful)"})
            span = year_span_from_frame(fd)
            if span is None:
                _empty("frontier")
            else:
                yr_min, yr_max = span
                pinned = st.session_state.get(SEASON_YEAR)
                default_range = (yr_min, yr_max)
                if pinned is not None and yr_min <= int(pinned) <= yr_max:
                    default_range = (max(yr_min, int(pinned) - 2), min(yr_max, int(pinned)))
                yr_range = st.slider("Years", yr_min, yr_max, default_range, key="frontier_year_range")
                fd = fd[fd["year_id"].between(yr_range[0], yr_range[1])]
                n_above = int(fd["above_frontier"].sum()) if "above_frontier" in fd.columns else 0
                st.caption(f"{n_above:,} of {len(fd):,} team-seasons above the efficiency frontier")

                panel_head("Payroll vs wins", "Dashed line is the polynomial envelope")
                fig = px.scatter(
                    fd,
                    x="payroll_m",
                    y="wins",
                    color="above_label",
                    hover_name="team_name",
                    hover_data=["year_id"],
                    labels={"payroll_m": "Payroll ($M)", "wins": "Wins", "above_label": "Status"},
                    color_discrete_map={"Above (efficient)": GREEN, "Below (wasteful)": CRIMSON},
                )
                if "frontier_pred" in fd.columns:
                    fl = fd.sort_values("payroll_m")[["payroll_m", "frontier_pred"]].drop_duplicates()
                    fig.add_trace(go.Scatter(
                        x=fl["payroll_m"],
                        y=fl["frontier_pred"],
                        mode="lines",
                        line=dict(color=CYAN, dash="dash", width=2),
                        name="Frontier",
                    ))
                fig.update_traces(marker=_SCATTER_MARKER, selector=dict(mode="markers"))
                _chart(fig, height=480)

                table_cols = [c for c in ["year_id", "team_name", "payroll_m", "wins", "frontier_pred", "above_frontier", "above_label"] if c in fd.columns]
                ef_col_cfg = {
                    "year_id": st.column_config.NumberColumn("Year", format="%d", width="small"),
                    "team_name": st.column_config.TextColumn("Team", width="medium"),
                    "payroll_m": st.column_config.NumberColumn("Payroll", format="$%.1fM"),
                    "wins": st.column_config.NumberColumn("Wins", format="%d"),
                    "frontier_pred": st.column_config.NumberColumn("Frontier", format="%.1f"),
                    "above_frontier": st.column_config.CheckboxColumn("Above"),
                    "above_label": st.column_config.TextColumn("Status"),
                }
                _show_table(fd[table_cols].sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True), ef_col_cfg, height=500)

    with cluster_tab:
        if clusters is None:
            _empty("clusters")
        else:
            cluster_summ = load_cluster_summary()
            if cluster_summ is not None:
                panel_head("Archetype summary")
                _show_table(cluster_summ, height=250)
            panel_head("Team-season cluster assignments")
            clust_cols = [c for c in ["year_id", "team_name", "cluster_label", "wins", "payroll", "team_total_war", "wins_per_10m", "window_phase"] if c in clusters.columns]
            clust_cfg = {**_TEAM_COL_CFG, "cluster_label": st.column_config.TextColumn("Archetype")}
            _show_table(
                scale_money_columns(clusters[clust_cols]).sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True),
                clust_cfg,
                height=500,
            )
            plot = add_payroll_millions(clusters.dropna(subset=["payroll", "wins"]))
            if not plot.empty:
                fig = px.scatter(
                    plot,
                    x="payroll_m",
                    y="wins",
                    color="cluster_label",
                    hover_name="team_name",
                    hover_data=["year_id"],
                    labels={"payroll_m": "Payroll ($M)", "wins": "Wins", "cluster_label": "Archetype"},
                )
                fig.update_traces(marker=_SCATTER_MARKER)
                _chart(fig, height=460)
