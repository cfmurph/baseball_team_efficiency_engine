"""
Baseball Team Efficiency Engine — Streamlit Dashboard

Sections:
  1. Overview       — Season-level efficiency scatter + summary KPIs
  2. Team Deep Dive — Single team: year-over-year trajectory, WAR, window phase
  3. Compare Teams  — Head-to-head comparison across any metrics
  4. Roster Lab     — Player WAR vs salary, contract labels, dead money
  5. Contract Watch — Worst contracts, surplus value leaders, dead money
  6. Efficiency Frontier — Payroll vs wins envelope; above/below curve
  7. What-If Sim    — Payroll redistribution scenario simulation
  8. Model Insights — Win model predictions, feature importance, misses
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLB Team Efficiency Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS = Path("artifacts")

_FILES = {
    "metrics":       ARTIFACTS / "team_onfield_contract_metrics.csv",
    "frontier":      ARTIFACTS / "team_efficiency_frontier.csv",
    "clusters":      ARTIFACTS / "team_clusters.csv",
    "cluster_summ":  ARTIFACTS / "team_cluster_summary.csv",
    "players":       ARTIFACTS / "player_season_metrics.csv",
    "top_value":     ARTIFACTS / "player_top_surplus_value.csv",
    "worst":         ARTIFACTS / "player_worst_contracts.csv",
    "dead":          ARTIFACTS / "player_dead_money.csv",
    "preds":         ARTIFACTS / "win_model_predictions.csv",
    "importance":    ARTIFACTS / "win_model_feature_importance.csv",
    "model_metrics": ARTIFACTS / "win_model_metrics.csv",
    "window":        ARTIFACTS / "team_window_phases.csv",
    "frontier_data": ARTIFACTS / "win_model_frontier_data.csv",
}


@st.cache_data(ttl=300)
def _load(key: str) -> pd.DataFrame | None:
    path = _FILES.get(key)
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _require(key: str, msg: str = "") -> pd.DataFrame:
    df = _load(key)
    if df is None:
        st.warning(f"Missing artifact: {_FILES[key].name}. {msg}")
        st.stop()
    return df


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
SECTIONS = [
    "Overview",
    "Team Deep Dive",
    "Compare Teams",
    "Roster Lab",
    "Contract Watch",
    "Efficiency Frontier",
    "What-If Sim",
    "Model Insights",
]

st.sidebar.title("⚾ MLB Efficiency Engine")
section = st.sidebar.radio("Navigate", SECTIONS)

metrics = _load("metrics")
if metrics is None:
    st.error("No artifacts found. Run the full pipeline first:\n\n```\npython -m pipeline.extract.pull_sources\npython -m pipeline.transform.build_warehouse\npython -m pipeline.transform.build_metrics\npython -m models.train_win_model\npython -m models.cluster_teams\n```")
    st.stop()

all_years = sorted(metrics["year_id"].dropna().unique().tolist())
all_teams = sorted(metrics["team_name"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
def section_overview() -> None:
    st.title("Overview — Season Efficiency")

    year = st.sidebar.slider("Season", int(all_years[0]), int(all_years[-1]), int(all_years[-1]))
    league_filter = st.sidebar.selectbox("League", ["Both", "AL", "NL"])

    season = metrics[metrics["year_id"] == year].copy()
    if league_filter != "Both" and "league_id" in season.columns:
        season = season[season["league_id"] == league_filter]

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Teams", len(season))
    c2.metric("Avg Payroll", f"${season['payroll'].mean() / 1e6:.1f}M")
    c3.metric("Avg Wins", f"{season['wins'].mean():.1f}")
    c4.metric("Avg Wins/$10M", f"{season['wins_per_10m'].mean():.2f}")
    war_col = "team_total_war"
    if war_col in season.columns:
        c5.metric("Avg Team WAR", f"{season[war_col].mean():.1f}")
    else:
        c5.metric("Avg Run Diff", f"{season['run_diff'].mean():.0f}")

    # Scatter
    color_col = "window_phase" if "window_phase" in season.columns else "league_id"
    size_col = "run_diff" if "run_diff" in season.columns else None
    size_arg = dict(size=size_col, size_max=28) if size_col else {}

    fig = px.scatter(
        season.dropna(subset=["payroll", "wins"]),
        x="payroll",
        y="wins",
        color=color_col,
        hover_name="team_name",
        hover_data=["payroll_per_win", "wins_per_10m"] + (["team_total_war"] if "team_total_war" in season.columns else []),
        labels={"payroll": "Payroll ($)", "wins": "Wins"},
        title=f"{year} — Payroll vs Wins",
        **size_arg,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # League ranking table
    st.subheader(f"{year} Team Rankings")
    display_cols = ["team_name", "wins", "run_diff", "payroll", "wins_per_10m", "payroll_per_win"]
    if "team_total_war" in season.columns:
        display_cols += ["team_total_war", "cost_per_war"]
    display_cols = [c for c in display_cols if c in season.columns]
    st.dataframe(
        season[display_cols].sort_values("wins", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 2. Team Deep Dive
# ---------------------------------------------------------------------------
def section_team_deep_dive() -> None:
    st.title("Team Deep Dive")

    team = st.sidebar.selectbox("Team", all_teams)
    team_df = metrics[metrics["team_name"] == team].sort_values("year_id")

    if team_df.empty:
        st.warning("No data for selected team.")
        return

    latest = team_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Season", int(latest["year_id"]))
    c2.metric("Wins", int(latest["wins"]))
    c3.metric("Payroll", f"${latest['payroll'] / 1e6:.1f}M")
    if "window_phase" in latest:
        c4.metric("Phase", str(latest["window_phase"]).title())

    # Wins over time
    fig_wins = px.line(team_df, x="year_id", y="wins", title=f"{team} — Wins Over Time", markers=True)
    if "pythag_wins" in team_df.columns:
        fig_wins.add_scatter(x=team_df["year_id"], y=team_df["pythag_wins"], mode="lines", name="Pythagorean W", line=dict(dash="dash"))
    fig_wins.update_layout(height=350)
    st.plotly_chart(fig_wins, use_container_width=True)

    col_left, col_right = st.columns(2)

    # Payroll trajectory
    with col_left:
        fig_pay = px.bar(team_df, x="year_id", y="payroll", title="Payroll ($)", color_discrete_sequence=["#2a7ae2"])
        fig_pay.update_layout(height=320)
        st.plotly_chart(fig_pay, use_container_width=True)

    # WAR / efficiency
    with col_right:
        if "team_total_war" in team_df.columns:
            fig_war = px.area(team_df, x="year_id", y="team_total_war", title="Team WAR", color_discrete_sequence=["#2ca02c"])
        else:
            fig_war = px.area(team_df, x="year_id", y="wins_per_10m", title="Wins per $10M", color_discrete_sequence=["#ff7f0e"])
        fig_war.update_layout(height=320)
        st.plotly_chart(fig_war, use_container_width=True)

    # Window phase timeline
    if "window_phase" in team_df.columns:
        st.subheader("Team Window Phase Timeline")
        phase_colors = {
            "contending": "#2ca02c",
            "developing": "#1f77b4",
            "steady": "#9467bd",
            "declining": "#ff7f0e",
            "rebuilding": "#d62728",
        }
        team_df["phase_color"] = team_df["window_phase"].map(phase_colors).fillna("#ccc")
        fig_phase = px.scatter(
            team_df,
            x="year_id",
            y="wins",
            color="window_phase",
            title="Phase Classification",
            color_discrete_map=phase_colors,
            size_max=14,
        )
        fig_phase.update_layout(height=280)
        st.plotly_chart(fig_phase, use_container_width=True)


# ---------------------------------------------------------------------------
# 3. Compare Teams
# ---------------------------------------------------------------------------
def section_compare_teams() -> None:
    st.title("Team Comparison")

    selected = st.sidebar.multiselect("Select Teams (2–5)", all_teams, default=all_teams[:3])
    if len(selected) < 2:
        st.info("Select at least 2 teams from the sidebar.")
        return

    year_range = st.sidebar.slider("Year range", int(all_years[0]), int(all_years[-1]), (int(all_years[-5]), int(all_years[-1])))
    compare_df = metrics[
        (metrics["team_name"].isin(selected)) &
        (metrics["year_id"] >= year_range[0]) &
        (metrics["year_id"] <= year_range[1])
    ].copy()

    metric_opts = [c for c in ["wins", "payroll", "wins_per_10m", "team_total_war", "cost_per_war", "surplus_value", "gini_salary", "run_diff"] if c in compare_df.columns]
    y_metric = st.selectbox("Metric", metric_opts, index=0)

    fig = px.line(compare_df, x="year_id", y=y_metric, color="team_name", markers=True, title=f"{y_metric} — {year_range[0]}–{year_range[1]}")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    # Summary table for latest year in range
    latest_year = compare_df["year_id"].max()
    latest = compare_df[compare_df["year_id"] == latest_year]
    show_cols = ["team_name", "wins", "payroll", "wins_per_10m", "payroll_per_win"]
    if "team_total_war" in latest.columns:
        show_cols += ["team_total_war", "cost_per_war", "surplus_value"]
    show_cols = [c for c in show_cols if c in latest.columns]
    st.subheader(f"Stats for {latest_year}")
    st.dataframe(latest[show_cols].sort_values("wins", ascending=False), use_container_width=True)


# ---------------------------------------------------------------------------
# 4. Roster Lab
# ---------------------------------------------------------------------------
def section_roster_lab() -> None:
    st.title("Roster Lab — Player WAR & Salary")

    players = _load("players")
    if players is None:
        st.warning("Player metrics not found. Run `python -m pipeline.transform.build_metrics`.")
        return

    col1, col2 = st.sidebar.columns(1), None
    team = st.sidebar.selectbox("Team", ["All"] + all_teams)
    yr_list = sorted(players["year_id"].dropna().unique().tolist()) if "year_id" in players.columns else []
    year = st.sidebar.selectbox("Year", yr_list, index=len(yr_list) - 1 if yr_list else 0)

    filt = players.copy()
    if team != "All" and "team_name" in filt.columns:
        filt = filt[filt["team_name"] == team]
    if year and "year_id" in filt.columns:
        filt = filt[filt["year_id"] == year]

    if filt.empty:
        st.info("No players match filter.")
        return

    # Scatter: WAR vs salary
    fig = px.scatter(
        filt.dropna(subset=["player_war", "salary"]),
        x="salary",
        y="player_war",
        color="contract_label" if "contract_label" in filt.columns else "player_type",
        hover_name="name_full" if "name_full" in filt.columns else "player_id",
        hover_data=["team_name", "year_id"] if "team_name" in filt.columns else ["year_id"],
        title=f"WAR vs Salary — {team} ({year})",
        labels={"salary": "Salary ($)", "player_war": "WAR"},
        color_discrete_map={
            "surplus_value": "#2ca02c",
            "fair_value": "#1f77b4",
            "overpaid": "#ff7f0e",
            "dead_money": "#d62728",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0 WAR")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    show_cols = ["name_full", "player_type", "player_war", "salary", "surplus_value", "contract_label"]
    show_cols = [c for c in show_cols if c in filt.columns]
    st.dataframe(
        filt[show_cols].sort_values("player_war", ascending=False).head(50).reset_index(drop=True),
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# 5. Contract Watch
# ---------------------------------------------------------------------------
def section_contract_watch() -> None:
    st.title("Contract Watch")

    tabs = st.tabs(["Top Surplus Value", "Worst Contracts", "Dead Money"])

    with tabs[0]:
        top = _load("top_value")
        if top is not None:
            st.subheader("Best-Value Players (highest surplus value)")
            display = ["name_full", "year_id", "team_name", "player_war", "salary", "surplus_value", "contract_label"]
            display = [c for c in display if c in top.columns]
            st.dataframe(top[display].head(50), use_container_width=True)

    with tabs[1]:
        worst = _load("worst")
        if worst is not None:
            st.subheader("Worst Contracts (most negative surplus value)")
            display = ["name_full", "year_id", "team_name", "player_war", "salary", "surplus_value", "contract_label"]
            display = [c for c in display if c in worst.columns]
            st.dataframe(worst[display].head(50), use_container_width=True)

    with tabs[2]:
        dead = _load("dead")
        if dead is not None:
            st.subheader("Dead Money (WAR ≤ 0, salary > 0)")
            display = ["name_full", "year_id", "team_name", "player_war", "salary", "surplus_value"]
            display = [c for c in display if c in dead.columns]
            st.dataframe(dead[display].head(50), use_container_width=True)


# ---------------------------------------------------------------------------
# 6. Efficiency Frontier
# ---------------------------------------------------------------------------
def section_efficiency_frontier() -> None:
    st.title("Efficiency Frontier — Payroll vs Wins")

    frontier_data = _load("frontier_data")
    clusters = _load("clusters")

    if frontier_data is not None:
        fd = frontier_data.copy()
        fd["above_label"] = fd["above_frontier"].map({True: "Above (Efficient)", False: "Below (Wasteful)"})
        year_range = st.sidebar.slider(
            "Year range",
            int(fd["year_id"].min()), int(fd["year_id"].max()),
            (int(fd["year_id"].min()), int(fd["year_id"].max())),
        )
        fd = fd[(fd["year_id"] >= year_range[0]) & (fd["year_id"] <= year_range[1])]

        fig = px.scatter(
            fd,
            x="payroll_m",
            y="wins",
            color="above_label",
            hover_name="team_name",
            hover_data=["year_id"],
            labels={"payroll_m": "Payroll ($M)", "wins": "Wins"},
            title="Efficiency Frontier: Teams above curve get more wins per $ spent",
            color_discrete_map={"Above (Efficient)": "#2ca02c", "Below (Wasteful)": "#d62728"},
        )
        # Draw the frontier line
        if "frontier_pred" in fd.columns:
            frontier_line = fd.sort_values("payroll_m")[["payroll_m", "frontier_pred"]].drop_duplicates()
            fig.add_trace(go.Scatter(
                x=frontier_line["payroll_m"],
                y=frontier_line["frontier_pred"],
                mode="lines",
                line=dict(color="#1f77b4", dash="dash", width=2),
                name="Frontier (75th pct)",
            ))
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        n_above = fd["above_frontier"].sum()
        st.info(f"{n_above} of {len(fd)} team-seasons are **above the frontier** (efficient).")
    else:
        st.info("Frontier data not yet generated. Run `python -m models.train_win_model`.")

    # Cluster view
    if clusters is not None:
        st.subheader("Team Archetypes by Cluster")
        cluster_summ = _load("cluster_summ")
        if cluster_summ is not None:
            st.dataframe(cluster_summ, use_container_width=True)

        fig_clust = px.scatter(
            clusters.dropna(subset=["payroll", "wins"]),
            x="payroll",
            y="wins",
            color="cluster_label",
            hover_name="team_name",
            hover_data=["year_id"],
            title="Payroll vs Wins by Cluster Archetype",
        )
        fig_clust.update_layout(height=460)
        st.plotly_chart(fig_clust, use_container_width=True)


# ---------------------------------------------------------------------------
# 7. What-If Simulation
# ---------------------------------------------------------------------------
def section_whatif() -> None:
    st.title("What-If Simulation — Payroll Impact")
    st.caption("Estimate win change if a team increases payroll by a given amount.")

    team = st.sidebar.selectbox("Team", all_teams, key="wi_team")
    team_df = metrics[metrics["team_name"] == team].sort_values("year_id")
    if team_df.empty:
        st.warning("No data.")
        return

    latest = team_df.iloc[-1]
    current_payroll = latest.get("payroll", 0)
    current_wins = latest.get("wins", 0)
    current_war = latest.get("team_total_war", None)

    st.subheader(f"{team} — {int(latest['year_id'])} Baseline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Payroll", f"${current_payroll / 1e6:.1f}M")
    col2.metric("Current Wins", int(current_wins))
    if current_war is not None:
        col3.metric("Team WAR", f"{current_war:.1f}")

    st.divider()
    payroll_delta_m = st.slider("Payroll increase ($M)", 0, 100, 20, step=5)
    new_payroll = current_payroll + payroll_delta_m * 1_000_000

    # Simple projection: use league-wide regression slope (wins ~ payroll)
    valid = metrics.dropna(subset=["payroll", "wins"])
    if len(valid) > 10:
        from numpy.polynomial.polynomial import polyfit as pfit
        coeffs = np.polyfit(valid["payroll"].values, valid["wins"].values, 1)
        slope = coeffs[0]
        win_gain = slope * (payroll_delta_m * 1_000_000)
    else:
        win_gain = 0.0

    projected_wins = current_wins + win_gain
    st.subheader("Projection")
    c1, c2, c3 = st.columns(3)
    c1.metric("New Payroll", f"${new_payroll / 1e6:.1f}M", delta=f"+${payroll_delta_m}M")
    c2.metric("Projected Wins", f"{projected_wins:.1f}", delta=f"+{win_gain:.1f}")
    c3.metric("New Payroll/Win", f"${new_payroll / max(projected_wins, 1) / 1e6:.2f}M" if projected_wins > 0 else "N/A")

    st.caption("Projection is a league-wide linear regression. Actual outcomes depend on how payroll is deployed.")

    # Historical chart
    fig = px.line(team_df, x="year_id", y="wins", title=f"{team} Win History + Projection", markers=True)
    fig.add_scatter(
        x=[int(latest["year_id"]) + 1],
        y=[projected_wins],
        mode="markers+text",
        marker=dict(color="orange", size=12, symbol="star"),
        text=["Projected"],
        textposition="top center",
        name="Projection",
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# 8. Model Insights
# ---------------------------------------------------------------------------
def section_model_insights() -> None:
    st.title("Model Insights — Win Prediction")

    model_metrics = _load("model_metrics")
    importance = _load("importance")
    preds = _load("preds")

    if model_metrics is not None:
        st.subheader("Model Performance")
        st.dataframe(model_metrics, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        if importance is not None:
            st.subheader("Feature Importance (XGBoost)")
            fig_imp = px.bar(
                importance.head(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 15 Features",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=440, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_right:
        if preds is not None and "actual_wins" in preds.columns and "predicted_wins_xgb" in preds.columns:
            st.subheader("Actual vs Predicted (XGBoost)")
            fig_preds = px.scatter(
                preds,
                x="actual_wins",
                y="predicted_wins_xgb",
                hover_name="team_name",
                hover_data=["year_id", "absolute_error_xgb"],
                title="Actual vs Predicted Wins",
                labels={"actual_wins": "Actual Wins", "predicted_wins_xgb": "XGBoost Predicted"},
            )
            lo = preds[["actual_wins", "predicted_wins_xgb"]].min().min() - 2
            hi = preds[["actual_wins", "predicted_wins_xgb"]].max().max() + 2
            fig_preds.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="red"), name="Perfect"))
            fig_preds.update_layout(height=440)
            st.plotly_chart(fig_preds, use_container_width=True)

    if preds is not None:
        st.subheader("Largest Prediction Misses")
        err_col = "absolute_error_xgb" if "absolute_error_xgb" in preds.columns else preds.columns[-1]
        st.dataframe(preds.sort_values(err_col, ascending=False).head(20), use_container_width=True)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
_SECTIONS = {
    "Overview": section_overview,
    "Team Deep Dive": section_team_deep_dive,
    "Compare Teams": section_compare_teams,
    "Roster Lab": section_roster_lab,
    "Contract Watch": section_contract_watch,
    "Efficiency Frontier": section_efficiency_frontier,
    "What-If Sim": section_whatif,
    "Model Insights": section_model_insights,
}

_SECTIONS[section]()
