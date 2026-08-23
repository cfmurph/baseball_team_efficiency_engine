"""
MLB Team Efficiency Engine — Dashboard

Sections
--------
1  Overview             Season efficiency leaders, scatter, standings, phases
2  Team Deep Dive       Franchise KPIs, history, roster, trend charts
3  Compare Teams        Multi-team table + metric trends
4  Roster Lab           Player WAR vs salary, searchable tables
5  Contract Watch       Surplus / overpaid / dead money
6  Efficiency Frontier  Payroll-wins envelope + cluster archetypes
7  What-If Sim          Payroll impact projection
8  Model Insights       Accuracy, feature importance, prediction misses
"""
from __future__ import annotations

import datetime
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.baseball_analytics.dashboard_helpers import (
    apply_layout_and_render_chart,
    compute_slider_max,
)

from src.baseball_analytics.dashboard_utils import (
    apply_plotly_layout,
    calculate_slider_max,
    player_id_columns_for_duplicate_names,
    render_plotly_chart,
    scale_payroll_for_display,
)

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.helpers import (
    CONTRACT_COLORS,
    add_payroll_millions,
    apply_efficiency_labels,
    artifact_status,
    empty_state_copy,
    filter_season,
    format_money_millions,
    format_ratio,
    format_signed_int,
    format_war,
    metric_label,
    nav_labels,
    nav_page,
    overview_kpi_payload,
    rank_by_efficiency,
    salary_coverage_note,
    scale_money_columns,
    slider_bounds,
    teams_from_frame,
    top_n_by,
    year_span_label,
    years_from_frame,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Efficiency Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background-color: #0d1117; }
.block-container {
    padding-top: 1.35rem !important;
    padding-bottom: 3rem !important;
    max-width: 1440px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stRadio label {
    color: #b1bac4;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 0.35rem 0.15rem;
    line-height: 1.35;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #e6edf3; }
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio [data-checked="true"] + label {
    color: #f85149 !important;
    font-weight: 600;
}
[data-testid="stSidebar"] [role="radiogroup"] label:focus-visible,
:focus-visible {
    outline: 2px solid #58a6ff !important;
    outline-offset: 2px;
}

.sidebar-brand {
    padding: 0.35rem 0.25rem 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 0.75rem;
}
.sidebar-brand h1 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.3;
    border: none;
    padding: 0;
}
.sidebar-brand span { color: #f85149; }
.sidebar-brand small {
    display: block;
    color: #b1bac4;
    font-size: 0.72rem;
    margin-top: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sidebar-status {
    margin-top: 1.25rem;
    padding-top: 0.9rem;
    border-top: 1px solid #21262d;
    color: #b1bac4;
    font-size: 0.74rem;
    line-height: 1.5;
}
.sidebar-status strong { color: #e6edf3; font-weight: 600; }
.nav-group {
    color: #8b949e;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0.15rem 0 0.35rem;
}

/* Titles */
h1 {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.03em !important;
    border-bottom: 2px solid #bf1c20;
    padding-bottom: 0.45rem;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.1rem !important;
}
h3 {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #b1bac4 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em !important;
}
.page-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f85149;
    margin-bottom: 0.15rem;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #b1bac4 !important;
    font-size: 0.86rem !important;
    line-height: 1.45;
}

/* KPI cards */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    min-height: 92px;
}
[data-testid="stMetricLabel"] {
    color: #b1bac4 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}
[data-testid="stMetricValue"] {
    color: #e6edf3 !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    line-height: 1.25;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; font-weight: 600; }

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 8px;
    overflow: hidden;
}
[data-testid="stDataFrame"] th,
.dvn-scroller .col-header-cell {
    background-color: #161b22 !important;
    color: #b1bac4 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid #30363d !important;
}
[data-testid="stDataFrame"] td {
    background-color: #0d1117;
    color: #e6edf3;
    font-size: 0.86rem;
    border-bottom: 1px solid #161b22 !important;
    font-variant-numeric: tabular-nums;
}

/* Tabs / inputs / buttons */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid #30363d; }
[data-testid="stTabs"] [role="tab"] {
    color: #b1bac4 !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.05rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #e6edf3 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #f85149 !important;
    border-bottom-color: #bf1c20 !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
    font-size: 0.88rem !important;
}
.stButton > button {
    background: #21262d;
    color: #e6edf3;
    border: 1px solid #30363d;
    font-size: 0.86rem;
    font-weight: 600;
    border-radius: 6px;
    min-height: 2.4rem;
}
.stButton > button:hover { background: #30363d; border-color: #8b949e; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #bf1c20 !important;
}
[data-testid="stExpander"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    background: #161b22 !important;
}
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.86rem !important; }
hr { border-color: #21262d !important; margin: 0.9rem 0 !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #21262d !important;
    color: #e6edf3 !important;
    border-radius: 4px !important;
}

.empty-card {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 10px;
    padding: 1.35rem 1.5rem;
    margin: 0.75rem 0 1.25rem;
}
.empty-card h3 {
    color: #e6edf3 !important;
    text-transform: none !important;
    letter-spacing: -0.02em !important;
    font-size: 1.05rem !important;
    margin: 0 0 0.4rem;
}
.empty-card p { color: #b1bac4; font-size: 0.9rem; margin: 0 0 0.75rem; line-height: 1.5; }
.empty-card pre {
    background: #0d1117;
    color: #e6edf3;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 0.75rem 0.9rem;
    font-size: 0.78rem;
    overflow-x: auto;
    margin: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Artifact paths ─────────────────────────────────────────────────────────────
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
    "sr_players":    ARTIFACTS / "sr_player_season_metrics.csv",
    "sr_injuries":   ARTIFACTS / "sr_injuries.csv",
    "sr_tx":         ARTIFACTS / "sr_transactions.csv",
}


@st.cache_data(ttl=300)
def _load(key: str) -> pd.DataFrame | None:
    path = _FILES.get(key)
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


# ── Column config ──────────────────────────────────────────────────────────────
_TEAM_COL_CFG = {
    "rank":              st.column_config.NumberColumn("#", format="%d", width="small"),
    "team_name":         st.column_config.TextColumn("Team", width="medium"),
    "year_id":           st.column_config.NumberColumn("Year", format="%d", width="small"),
    "wins":              st.column_config.NumberColumn("W", format="%d", width="small"),
    "losses":            st.column_config.NumberColumn("L", format="%d", width="small"),
    "run_diff":          st.column_config.NumberColumn("Run Diff", format="%+d"),
    "payroll":           st.column_config.NumberColumn("Payroll ($M)", format="$%.1fM"),
    "payroll_per_win":   st.column_config.NumberColumn("$/Win ($M)", format="$%.2fM"),
    "wins_per_10m":      st.column_config.NumberColumn("W/$10M", format="%.2f"),
    "team_total_war":    st.column_config.NumberColumn("Team WAR", format="%.1f"),
    "war_source":        st.column_config.TextColumn("WAR source"),
    "cost_per_war":      st.column_config.NumberColumn("$/WAR ($M)", format="$%.2fM"),
    "war_per_1m":        st.column_config.NumberColumn("WAR/$1M", format="%.2f"),
    "surplus_value":     st.column_config.NumberColumn("Surplus ($M)", format="$%.1fM"),
    "pythag_wins":       st.column_config.NumberColumn("Pythag W", format="%.1f"),
    "pythag_gap":        st.column_config.NumberColumn("Pythag Gap", format="%+.1f"),
    "gini_salary":       st.column_config.NumberColumn("Gini", format="%.3f"),
    "dead_money_share":  st.column_config.NumberColumn("Dead Money %", format="%.1f%%"),
    "window_phase":      st.column_config.TextColumn("Phase"),
    "league_id":         st.column_config.TextColumn("Lg", width="small"),
    "efficiency_label":  st.column_config.TextColumn("Efficiency"),
}

_PLAYER_COL_CFG = {
    "name_full":       st.column_config.TextColumn("Player"),
    "year_id":         st.column_config.NumberColumn("Year", format="%d"),
    "team_name":       st.column_config.TextColumn("Team"),
    "player_type":     st.column_config.TextColumn("Type"),
    "primary_position": st.column_config.TextColumn("Pos"),
    "pa":              st.column_config.NumberColumn("PA", format="%d"),
    "hr":              st.column_config.NumberColumn("HR", format="%d"),
    "bb":              st.column_config.NumberColumn("BB", format="%d"),
    "woba":            st.column_config.NumberColumn("wOBA", format="%.3f"),
    "ip":              st.column_config.NumberColumn("IP", format="%.1f"),
    "era":             st.column_config.NumberColumn("ERA", format="%.2f"),
    "fip":             st.column_config.NumberColumn("FIP", format="%.2f"),
    "batting_war":     st.column_config.NumberColumn("bWAR", format="%.1f"),
    "pitching_war":    st.column_config.NumberColumn("pWAR", format="%.1f"),
    "player_war":      st.column_config.NumberColumn("WAR", format="%.1f"),
    "war_source":      st.column_config.TextColumn("WAR source"),
    "salary":          st.column_config.NumberColumn("Salary ($M)", format="$%.2fM"),
    "surplus_value":   st.column_config.NumberColumn("Surplus ($M)", format="$%.2fM"),
    "contract_label":  st.column_config.TextColumn("Contract"),
}


def _scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw $ to $M for display."""
    return scale_payroll_for_display(df)


def _show_table(df: pd.DataFrame, col_cfg: dict | None = None, height: int = 600, **kwargs) -> None:
    cfg = {k: v for k, v in (col_cfg or {}).items() if k in df.columns}
    kwargs.setdefault("hide_index", True)
    st.dataframe(df, column_config=cfg, use_container_width=True, height=height, **kwargs)


_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(family="Inter, -apple-system, sans-serif", color="#e6edf3", size=12),
    title_font=dict(size=14, color="#e6edf3", family="Inter, sans-serif"),
    xaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        tickfont=dict(color="#b1bac4", size=11),
        title_font=dict(color="#b1bac4", size=12),
    ),
    yaxis=dict(
        gridcolor="#21262d",
        linecolor="#30363d",
        tickcolor="#30363d",
        tickfont=dict(color="#b1bac4", size=11),
        title_font=dict(color="#b1bac4", size=12),
    ),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1, font=dict(size=11, color="#e6edf3")),
    margin=dict(t=48, b=36, l=16, r=16),
    colorway=["#f85149", "#58a6ff", "#3fb950", "#d29922", "#a371f7", "#f78166", "#1f6feb"],
)

_SCATTER_MARKER = dict(size=8, opacity=0.82, line=dict(width=0.5, color="#0d1117"))


def _apply_layout(fig) -> None:
    """Apply the Baseball Savant dark layout to any Plotly figure."""
    apply_plotly_layout(fig, _PLOTLY_LAYOUT)


def _chart(fig, height: int = 400) -> None:
    """Apply dark layout and render a Plotly chart."""
    apply_layout_and_render_chart(
        fig,
        apply_layout=_apply_layout,
        plotly_chart=st.plotly_chart,
        height=height,
    )


# ── Global state ───────────────────────────────────────────────────────────────
metrics = _load("metrics")
if metrics is None:
    st.error(
        "No artifacts found. Run the full pipeline first:\n\n"
        "```\npython3 -m pipeline.extract.pull_sources\n"
        "python3 -m pipeline.extract.pull_war\n"
        "python3 -m pipeline.transform.build_warehouse\n"
        "python3 -m pipeline.transform.build_metrics\n"
        "python3 -m models.train_win_model\n"
        "python3 -m models.cluster_teams\n```"
    )


def _salary_note(year: int | None) -> None:
    note = salary_coverage_note(year)
    if note:
        st.info(note)


# ── Global state ───────────────────────────────────────────────────────────────
metrics = _load("metrics")
_current_year = datetime.date.today().year
all_years = sorted(metrics["year_id"].dropna().astype(int).unique().tolist())
_slider_max = compute_slider_max(all_years, _current_year)
all_teams = sorted(metrics["team_name"].dropna().unique().tolist())


def _season_picker(key: str = "season", default_latest: bool = True) -> int | None:
    """Compact season selector: selectbox + previous/next buttons."""
    if not all_years:
        return None
    default_idx = len(all_years) - 1 if default_latest else 0
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.button("◀", key=f"{key}_prev", help="Previous season"):
            st.session_state[f"{key}_idx"] = max(0, st.session_state.get(f"{key}_idx", default_idx) - 1)
    with c3:
        if st.button("▶", key=f"{key}_next", help="Next season"):
            st.session_state[f"{key}_idx"] = min(len(all_years) - 1, st.session_state.get(f"{key}_idx", default_idx) + 1)
    idx = st.session_state.get(f"{key}_idx", default_idx)
    with c2:
        chosen = st.selectbox("Season", all_years, index=idx, key=f"{key}_sel", label_visibility="collapsed")
        st.session_state[f"{key}_idx"] = all_years.index(chosen)
    return int(chosen)


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <h1>⚾ MLB <span>Efficiency</span></h1>
      <small>Team &amp; player analytics</small>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown('<div class="nav-group">Sections</div>', unsafe_allow_html=True)
page = st.sidebar.radio(
    "Dashboard section",
    nav_labels(),
    label_visibility="collapsed",
)
missing_core = [key for key in ("metrics", "players", "frontier_data", "preds") if key in _status["missing"]]
status_bits = [
    f"<strong>Seasons</strong> {html.escape(year_span_label(all_years))}",
    f"<strong>Artifacts</strong> {_status['n_present']}/{_status['n_total']}",
]
if missing_core:
    status_bits.append("Core files missing — pages show setup steps.")
else:
    status_bits.append("Payroll metrics are fullest for 1990–2016.")
st.sidebar.markdown(
    '<div class="sidebar-status">' + "<br/>".join(status_bits) + "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_league_snapshot() -> None:
    st.title("League Snapshot")
    st.caption(
        "Full sortable team table for any season. Team WAR is Baseball-Reference rWAR "
        "rolled up from players (`war_source=real`); Lahman wOBA/FIP approx is the fallback."
    )

    col_nav, col_lg = st.columns([3, 1])
    with col_nav:
        year = _season_picker("snap")
    with col_lg:
        lg = st.selectbox("League", ["All", "AL", "NL"], key="snap_lg")
    if year is None:
        _empty("season")
        return

    season = apply_efficiency_labels(filter_season(metrics, year, lg))
    _salary_note(year)

    if season.empty:
        _empty("season")
        return

    cards = overview_kpi_payload(season)
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.metric(card["label"], card["value"], delta=card["delta"])

    ranked = rank_by_efficiency(season)
    extra = tuple(c for c in ("wins", "payroll", "team_total_war", "wins_per_10m") if c in season.columns)
    cheap = top_n_by(season, "surplus_value", n=5, extra_cols=extra)
    dear = top_n_by(season, "surplus_value", n=5, ascending=True, extra_cols=extra)

    if not cheap.empty:
        left, right = st.columns(2)
        with left:
            st.subheader("Buying wins cheaply")
            st.caption("Highest surplus value — WAR produced above market payroll.")
            _show_table(scale_money_columns(cheap), _TEAM_COL_CFG, height=220)
        with right:
            st.subheader("Paying above market")
            st.caption("Lowest surplus value — expensive relative to on-field WAR.")
            _show_table(scale_money_columns(dear), _TEAM_COL_CFG, height=220)

    plot_df = add_payroll_millions(season.dropna(subset=["payroll", "wins"]))
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
            title=f"{year} — Payroll vs wins",
            color_continuous_scale=["#f85149", "#d29922", "#3fb950"] if color_col == "surplus_m" else None,
        )
        fig.update_traces(marker=_SCATTER_MARKER)
        fig.update_layout(coloraxis_colorbar_title="Surplus ($M)" if color_col == "surplus_m" else None)
        _chart(fig, height=460)
    else:
        st.caption("No payroll values to plot for this season.")

    rank_tab, standings_tab, phase_tab = st.tabs(["Efficiency ranking", "Standings", "Window phases"])

    table_cols = [
        "rank", "team_name", "league_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "war_source", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ]
    with rank_tab:
        st.caption("Sorted by surplus value, then wins per $10M. Click a header to re-sort.")
        display_cols = [c for c in table_cols if c in ranked.columns]
        _show_table(scale_money_columns(ranked[display_cols]), _TEAM_COL_CFG, height=560)

    with standings_tab:
        leagues = [value for value in ("AL", "NL") if "league_id" in season.columns and (season["league_id"] == value).any()]
        if not leagues:
            _show_table(
                scale_money_columns(ranked[[c for c in ["rank", "team_name", "wins", "losses", "run_diff", "window_phase"] if c in ranked.columns]]),
                _TEAM_COL_CFG,
                height=420,
            )
        else:
            cols = st.columns(len(leagues))
            stand_cols = [c for c in ["team_name", "wins", "losses", "run_diff", "payroll", "wins_per_10m", "team_total_war", "window_phase"] if c in season.columns]
            for col, lg_name in zip(cols, leagues):
                with col:
                    st.subheader(lg_name)
                    lg_df = season[season["league_id"] == lg_name].sort_values("wins", ascending=False)
                    _show_table(scale_money_columns(lg_df[stand_cols]).reset_index(drop=True), _TEAM_COL_CFG, height=360)

    with phase_tab:
        window_df = _load("window")
        if window_df is None:
            if "window_phase" not in season.columns:
                _empty("window")
            else:
                phases = ["All"] + sorted(season["window_phase"].dropna().astype(str).unique().tolist())
                phase_filter = st.selectbox("Filter by phase", phases, key="ov_phase_season")
                phase_view = season if phase_filter == "All" else season[season["window_phase"] == phase_filter]
                cols = [c for c in ["team_name", "wins", "payroll", "team_total_war", "window_phase"] if c in phase_view.columns]
                _show_table(scale_money_columns(phase_view[cols]).sort_values("wins", ascending=False).reset_index(drop=True), _TEAM_COL_CFG, height=420)
        else:
            display = window_df.copy()
            if "payroll" in display.columns:
                display["payroll"] = display["payroll"] / 1_000_000
            phases = ["All"] + sorted(display["window_phase"].dropna().astype(str).unique().tolist()) if "window_phase" in display.columns else ["All"]
            phase_filter = st.selectbox("Filter by phase", phases, key="ov_phase")
            if phase_filter != "All":
                display = display[display["window_phase"] == phase_filter]
            phase_cfg = {
                **_TEAM_COL_CFG,
                "year_id": st.column_config.NumberColumn("Latest Year", format="%d"),
            }
            _show_table(display.sort_values("wins", ascending=False).reset_index(drop=True), phase_cfg, height=480)


# ══════════════════════════════════════════════════════════════════════════════
# 2. ROSTER LAB
# ══════════════════════════════════════════════════════════════════════════════
def page_player_explorer() -> None:
    st.title("Player Explorer")
    st.caption(
        "All player stats for any season. WAR is Baseball-Reference rWAR when the "
        "player-season maps (`war_source=real`); otherwise the Lahman approximation."
    )

    players = _load("players")
    sr_players = _load("sr_players")
    if players is None:
        _empty("players")
        return

    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
    with f1:
        yr_opts = years_from_frame(players)
        year = st.selectbox("Season", yr_opts, index=len(yr_opts) - 1 if yr_opts else 0, key="pe_year")
    with f2:
        team_opts = ["All Teams"] + teams_from_frame(players)
        team = st.selectbox("Team", team_opts, key="pe_team")
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
            fig = px.scatter(
                plot_f,
                x="salary",
                y="player_war",
                color="contract_label" if "contract_label" in plot_f.columns else None,
                hover_name="name_full" if "name_full" in plot_f.columns else None,
                hover_data=[c for c in ["team_name", "year_id"] if c in plot_f.columns],
                labels={"salary": "Salary ($M)", "player_war": "WAR", "contract_label": "Contract"},
                color_discrete_map=CONTRACT_COLORS,
                title="WAR vs salary",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#30363d")
            fig.update_traces(marker=_SCATTER_MARKER)
            _chart(fig, height=420)

    # Detect same-name players in the current filtered view so we can show player_id
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
        st.subheader("Sportradar stats (real WAR · wRC+ · ERA-)")
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEAM DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
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
        team = st.selectbox("Team", all_teams, key="tp_team")
    with c2:
        year = _season_picker("tp")

    team_history = metrics[metrics["team_name"] == team].sort_values("year_id")
    if team_history.empty:
        _empty("team")
        return

    season_row = team_history[team_history["year_id"] == year] if year is not None else team_history.iloc[0:0]
    st.subheader(f"{team} — {year if year is not None else '—'}")
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

    st.divider()
    st.subheader("Season history")
    hist_cols = [c for c in [
        "year_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "war_source", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ] if c in team_history.columns]
    _show_table(
        scale_money_columns(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True),
        _TEAM_COL_CFG,
        height=400,
    )

    with st.expander("Trend charts", expanded=True):
        ch1, ch2, ch3 = st.columns(3)
        with ch1:
            fig_w = px.line(team_history, x="year_id", y="wins", markers=True, title="Wins")
            if "pythag_wins" in team_history.columns:
                fig_w.add_scatter(
                    x=team_history["year_id"],
                    y=team_history["pythag_wins"],
                    mode="lines",
                    name="Pythag W",
                    line=dict(dash="dash", color="#8b949e"),
                )
            fig_w.update_layout(xaxis_title="Season", yaxis_title="Wins")
            _chart(fig_w, height=280)
        with ch2:
            if team_history["payroll"].notna().any():
                pay = add_payroll_millions(team_history)
                fig_p = px.bar(pay, x="year_id", y="payroll_m", title="Payroll ($M)", color_discrete_sequence=["#f85149"])
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
                _chart(fig_ph, height=280)

    st.subheader(f"Roster — {year if year is not None else '—'}")
    players = _load("players")
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


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPARE TEAMS
# ══════════════════════════════════════════════════════════════════════════════
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
        selected = st.multiselect("Teams", all_teams, default=all_teams[:4], key="sc_teams")
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
    st.subheader(f"Latest season in range — {latest_year}")
    latest = compare_df[compare_df["year_id"] == latest_year]
    table_cols = [c for c in [
        "team_name", "wins", "losses", "run_diff", "pythag_wins",
        "payroll", "wins_per_10m", "team_total_war", "cost_per_war",
        "surplus_value", "gini_salary", "window_phase",
    ] if c in latest.columns]
    _show_table(scale_money_columns(latest[table_cols]).sort_values("wins", ascending=False).reset_index(drop=True), _TEAM_COL_CFG, height=250)

    st.subheader(f"History — {year_range[0]}–{year_range[1]}")
    hist_cols = [c for c in [
        "year_id", "team_name", "wins", "run_diff", "payroll",
        "wins_per_10m", "team_total_war", "surplus_value", "window_phase",
    ] if c in compare_df.columns]
    _show_table(
        scale_money_columns(compare_df[hist_cols]).sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True),
        _TEAM_COL_CFG,
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


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONTRACT WATCH
# ══════════════════════════════════════════════════════════════════════════════
def page_contract_analysis() -> None:
    st.title("Contract Analysis")
    st.caption(
        "Every player contract, classified and searchable. Salary data from Lahman (through 2016). "
        "Surplus value uses Baseball-Reference rWAR when `war_source=real`."
    )

    players = _load("players")
    if players is None:
        _empty("players")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        yr_opts = years_from_frame(players)
        year = st.selectbox("Season", ["All Seasons"] + yr_opts, key="ca_year")
    with f2:
        team = st.selectbox("Team", ["All Teams"] + teams_from_frame(players), key="ca_team")
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
        _show_table(display, _PLAYER_COL_CFG)

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
        fig = px.scatter(
            plot_f,
            x="salary",
            y="player_war",
            color="contract_label" if "contract_label" in plot_f.columns else None,
            hover_name="name_full" if "name_full" in plot_f.columns else None,
            hover_data=[c for c in ["year_id", "team_name"] if c in plot_f.columns],
            labels={"salary": "Salary ($M)", "player_war": "WAR", "contract_label": "Contract"},
            color_discrete_map=CONTRACT_COLORS,
            title="WAR vs salary",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#30363d")
        fig.update_traces(marker=_SCATTER_MARKER)
        _chart(fig, height=450)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EFFICIENCY FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
def page_efficiency_frontier() -> None:
    _page_header("Efficiency Frontier")
    frontier_data = _load("frontier_data")
    clusters = _load("clusters")
    frontier_tab, cluster_tab = st.tabs(["Frontier", "Team Archetypes"])

    with frontier_tab:
        if frontier_data is None:
            _empty("frontier")
        else:
            fd = frontier_data.copy()
            fd["above_label"] = fd["above_frontier"].map({True: "Above (efficient)", False: "Below (wasteful)"})
            yr_min, yr_max = int(fd["year_id"].min()), max(int(fd["year_id"].max()), _current_year)
            yr_range = st.slider("Years", yr_min, yr_max, (yr_min, yr_max), key="ef_range")
            fd = fd[fd["year_id"].between(yr_range[0], yr_range[1])]
            n_above = int(fd["above_frontier"].sum()) if "above_frontier" in fd.columns else 0
            st.caption(f"{n_above:,} of {len(fd):,} team-seasons above the efficiency frontier")

            fig = px.scatter(
                fd,
                x="payroll_m",
                y="wins",
                color="above_label",
                hover_name="team_name",
                hover_data=["year_id"],
                labels={"payroll_m": "Payroll ($M)", "wins": "Wins", "above_label": "Status"},
                color_discrete_map={"Above (efficient)": "#3fb950", "Below (wasteful)": "#f85149"},
                title="Payroll vs wins with frontier envelope",
            )
            if "frontier_pred" in fd.columns:
                fl = fd.sort_values("payroll_m")[["payroll_m", "frontier_pred"]].drop_duplicates()
                fig.add_trace(go.Scatter(
                    x=fl["payroll_m"],
                    y=fl["frontier_pred"],
                    mode="lines",
                    line=dict(color="#58a6ff", dash="dash", width=2),
                    name="Frontier",
                ))
            fig.update_traces(marker=_SCATTER_MARKER, selector=dict(mode="markers"))
            _chart(fig, height=480)

            table_cols = [c for c in ["year_id", "team_name", "payroll_m", "wins", "frontier_pred", "above_frontier", "above_label"] if c in fd.columns]
            ef_col_cfg = {
                "year_id": st.column_config.NumberColumn("Year", format="%d", width="small"),
                "team_name": st.column_config.TextColumn("Team", width="medium"),
                "payroll_m": st.column_config.NumberColumn("Payroll ($M)", format="$%.1fM"),
                "wins": st.column_config.NumberColumn("Wins", format="%d"),
                "frontier_pred": st.column_config.NumberColumn("Frontier Pred", format="%.1f"),
                "above_frontier": st.column_config.CheckboxColumn("Above Curve"),
                "above_label": st.column_config.TextColumn("Status"),
            }
            _show_table(fd[table_cols].sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True), ef_col_cfg, height=500)

    with cluster_tab:
        if clusters is None:
            _empty("clusters")
        else:
            cluster_summ = _load("cluster_summ")
            if cluster_summ is not None:
                st.subheader("Archetype summary")
                _show_table(cluster_summ, height=250)
            st.subheader("Team-season cluster assignments")
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
                    title="Clusters by payroll and wins",
                )
                fig.update_traces(marker=_SCATTER_MARKER)
                _chart(fig, height=460)


# ══════════════════════════════════════════════════════════════════════════════
# 7. WHAT-IF SIM
# ══════════════════════════════════════════════════════════════════════════════
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
        team = st.selectbox("Team", all_teams, key="wi_team")
    with c2:
        year = _season_picker("wi")

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

    st.subheader(f"{team} — {int(r['year_id'])} baseline")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Payroll", format_money_millions(current_payroll))
    k2.metric("Wins", int(current_wins))
    k3.metric("W/$10M", format_ratio(r.get("wins_per_10m")))
    k4.metric("Team WAR", format_war(current_war))

    st.divider()
    payroll_delta_m = st.slider("Payroll change ($M)", -50, 150, 20, step=5, key="wi_delta")
    new_payroll = current_payroll + payroll_delta_m * 1_000_000
    valid = metrics.dropna(subset=["payroll", "wins"])
    if len(valid) > 10:
        coeffs = np.polyfit(valid["payroll"].values, valid["wins"].values, 1)
        win_gain = coeffs[0] * (payroll_delta_m * 1_000_000)
    else:
        win_gain = 0.0
    projected_wins = current_wins + win_gain

    k1, k2, k3 = st.columns(3)
    k1.metric("New payroll", format_money_millions(new_payroll), delta=f"{payroll_delta_m:+.0f}M")
    k2.metric("Projected wins", f"{projected_wins:.0f}", delta=f"{win_gain:+.1f}")
    k3.metric("New $/win", format_money_millions(new_payroll / max(projected_wins, 1), decimals=2) if projected_wins > 0 else "—")
    st.caption("Linear regression on all historical team-seasons. Actual results depend on how the extra payroll is allocated.")

    st.subheader("Historical record")
    hist_cols = [c for c in ["year_id", "wins", "run_diff", "payroll", "wins_per_10m", "team_total_war", "window_phase"] if c in team_history.columns]
    _show_table(
        scale_money_columns(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True),
        _TEAM_COL_CFG,
        height=350,
    )
    fig = px.line(team_history, x="year_id", y="wins", markers=True, title=f"{team} — win history")
    fig.add_scatter(
        x=[int(r["year_id"]) + 1],
        y=[projected_wins],
        mode="markers+text",
        marker=dict(color="#d29922", size=14, symbol="star"),
        text=["Projected"],
        textposition="top center",
        name="Projection",
    )
    fig.update_layout(xaxis_title="Season", yaxis_title="Wins")
    _chart(fig, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_model_insights() -> None:
    _page_header("Model Insights")
    model_metrics_df = _load("model_metrics")
    importance = _load("importance")
    preds = _load("preds")
    if model_metrics_df is None and importance is None and preds is None:
        _empty("models")
        return

    perf_tab, feat_tab, pred_tab = st.tabs(["Performance", "Feature Importance", "Predictions"])

    with perf_tab:
        if model_metrics_df is None:
            _empty("models")
        else:
            cards = st.columns(min(len(model_metrics_df), 4) or 1)
            for col, (_, row) in zip(cards, model_metrics_df.iterrows()):
                name = str(row.get("model", "Model"))
                mae = row.get("mae")
                r2 = row.get("r2")
                col.metric(name, f"MAE {mae:.2f}" if pd.notna(mae) else "—", delta=f"R² {r2:.3f}" if pd.notna(r2) else None)
            cfg = {
                "model": st.column_config.TextColumn("Model"),
                "mae": st.column_config.NumberColumn("MAE (wins)", format="%.2f"),
                "r2": st.column_config.NumberColumn("R²", format="%.4f"),
                "n_rows": st.column_config.NumberColumn("N", format="%d"),
            }
            _show_table(model_metrics_df, cfg, height=180)

    with feat_tab:
        if importance is None:
            _empty("models")
        else:
            cfg = {
                "feature": st.column_config.TextColumn("Feature", width="medium"),
                "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
            }
            ranked = importance.sort_values("importance", ascending=False).reset_index(drop=True)
            _show_table(ranked, cfg, height=360)
            fig = px.bar(
                ranked.head(15),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#21262d"], [1, "#f85149"]],
                title="Top 15 features",
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            _chart(fig, height=420)

    with pred_tab:
        if preds is None:
            _empty("models")
        else:
            err_col = "absolute_error_xgb" if "absolute_error_xgb" in preds.columns else (
                "absolute_error_lr" if "absolute_error_lr" in preds.columns else None
            )
            pred_cfg = {
                "team_name": st.column_config.TextColumn("Team", width="medium"),
                "year_id": st.column_config.NumberColumn("Year", format="%d", width="small"),
                "actual_wins": st.column_config.NumberColumn("Actual W", format="%d"),
                "predicted_wins_xgb": st.column_config.NumberColumn("XGB Pred", format="%.1f"),
                "predicted_wins_lr": st.column_config.NumberColumn("LR Pred", format="%.1f"),
                "absolute_error_xgb": st.column_config.NumberColumn("XGB Error", format="%.1f"),
                "absolute_error_lr": st.column_config.NumberColumn("LR Error", format="%.1f"),
            }
            sort_df = preds.sort_values(err_col, ascending=False).reset_index(drop=True) if err_col else preds
            st.caption(f"{len(sort_df):,} predictions — sorted by largest absolute error")
            _show_table(sort_df, pred_cfg, height=480)
            if "actual_wins" in preds.columns and "predicted_wins_xgb" in preds.columns:
                fig = px.scatter(
                    preds,
                    x="actual_wins",
                    y="predicted_wins_xgb",
                    hover_name="team_name",
                    hover_data=["year_id"],
                    labels={"actual_wins": "Actual wins", "predicted_wins_xgb": "XGB predicted"},
                    title="Actual vs predicted wins",
                )
                lo = preds[["actual_wins", "predicted_wins_xgb"]].min().min() - 2
                hi = preds[["actual_wins", "predicted_wins_xgb"]].max().max() + 2
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="#30363d"), name="Perfect"))
                fig.update_traces(marker=_SCATTER_MARKER, selector=dict(mode="markers"))
                _chart(fig, height=400)


# ── Routing ────────────────────────────────────────────────────────────────────
_PAGES = {
    "Overview": page_overview,
    "Team Deep Dive": page_team_deep_dive,
    "Compare Teams": page_compare_teams,
    "Roster Lab": page_roster_lab,
    "Contract Watch": page_contract_watch,
    "Efficiency Frontier": page_efficiency_frontier,
    "What-If Sim": page_whatif,
    "Model Insights": page_model_insights,
}

_PAGES[page]()
