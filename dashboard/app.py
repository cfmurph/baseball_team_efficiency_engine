from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ARTIFACTS = Path("artifacts")
METRICS_PATH = ARTIFACTS / "team_onfield_contract_metrics.csv"
CLUSTERS_PATH = ARTIFACTS / "team_clusters.csv"
PREDS_PATH = ARTIFACTS / "win_model_predictions.csv"

st.set_page_config(page_title="Baseball Team Efficiency Engine", layout="wide")
st.title("Baseball Team Efficiency Engine")
st.caption("Performance, payroll efficiency, and team archetype analysis")

if not METRICS_PATH.exists():
    st.warning("Run the pipeline first so artifacts exist.")
    st.stop()

metrics = pd.read_csv(METRICS_PATH)
clusters = pd.read_csv(CLUSTERS_PATH) if CLUSTERS_PATH.exists() else pd.DataFrame()
preds = pd.read_csv(PREDS_PATH) if PREDS_PATH.exists() else pd.DataFrame()

min_year = int(metrics["year_id"].min())
max_year = int(metrics["year_id"].max())
year = st.sidebar.slider("Season", min_year, max_year, max_year)
team_options = sorted(metrics["team_name"].dropna().unique().tolist())
selected_teams = st.sidebar.multiselect("Teams", team_options, default=team_options[:2])

season_df = metrics[metrics["year_id"] == year].copy()
if selected_teams:
    compare_df = season_df[season_df["team_name"].isin(selected_teams)]
else:
    compare_df = season_df.copy()

col1, col2, col3 = st.columns(3)
col1.metric("Avg Payroll", f"${season_df['payroll'].mean():,.0f}")
col2.metric("Avg Wins", f"{season_df['wins'].mean():.1f}")
col3.metric("Avg Wins per $10M", f"{season_df['wins_per_10m'].mean():.2f}")

fig = px.scatter(
    season_df,
    x="payroll",
    y="wins",
    size="run_diff",
    hover_name="team_name",
    title=f"Payroll vs Wins ({year})",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Team Comparison")
st.dataframe(compare_df[[
    "team_name", "wins", "run_diff", "pythag_wins", "payroll",
    "top_1_salary_share", "top_5_salary_share", "gini_salary",
    "payroll_per_win", "wins_per_10m"
]].sort_values("wins", ascending=False), use_container_width=True)

if selected_teams:
    trend_df = metrics[metrics["team_name"].isin(selected_teams)].copy()
    trend = px.line(trend_df, x="year_id", y="wins", color="team_name", title="Wins Over Time")
    st.plotly_chart(trend, use_container_width=True)

if not clusters.empty:
    st.subheader("Cluster View")
    cluster_year_df = clusters[clusters["year_id"] == year].copy()
    cluster_fig = px.scatter(cluster_year_df, x="payroll", y="wins", color="cluster_label", hover_name="team_name")
    st.plotly_chart(cluster_fig, use_container_width=True)

if not preds.empty:
    st.subheader("Largest Win Model Misses")
    st.dataframe(preds.sort_values("absolute_error", ascending=False).head(15), use_container_width=True)
