"""
MLB Team Efficiency Engine — Front-office dashboard

Entrypoint: ``streamlit run dashboard/app.py`` (from repo root).

Pages call named loaders in ``dashboard.data`` — never raw artifact paths.
Shared session keys (``dashboard.state``): season_year, selected_team,
selected_league. Artifact resolution uses ``resolve_artifact()``
(ARTIFACTS_URI → local artifacts/ fallback).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st
from src.baseball_analytics.config import load_artifact_settings
from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_duplicate_names,
    scale_payroll_for_display,
)
from src.baseball_analytics.storage import (
    artifact_source_label,
    resolve_artifact,
)
from dashboard.helpers import (
    artifact_status,
    nav_labels,
    teams_from_frame,
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
# Shared storage (ARTIFACTS_URI) is preferred; local artifacts/ is the fallback.
# See docs/shared_artifacts.md.
_ARTIFACT_SETTINGS = load_artifact_settings(str(_ROOT / "config/settings.yaml"))
ARTIFACTS = _ARTIFACT_SETTINGS.local_dir
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
def _resolve_file(key: str) -> str | None:
    path = _FILES.get(key)
    if path is None:
        return None
    resolved = resolve_artifact(path.name, _ARTIFACT_SETTINGS)
    return None if resolved is None else str(resolved)


@st.cache_data(ttl=300)
def _load(key: str) -> pd.DataFrame | None:
    raw = _resolve_file(key)
    if raw is None:
        return None
    return pd.read_csv(raw)


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
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, Plus Jakarta Sans, sans-serif", color="#94a3b8", size=12),
    margin=dict(t=56, b=40, l=48, r=20),
)
_SCATTER_MARKER = ui.SCATTER_MARKER


def _apply_layout(fig) -> None:
    fig.update_layout(**_PLOTLY_LAYOUT)


def _chart(fig, height: int = 400) -> None:
    _apply_layout(fig)
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


_ARTIFACT_SETTINGS = load_artifact_settings(str(_ROOT / "config/settings.yaml"))
ARTIFACTS = _ARTIFACT_SETTINGS.local_dir
_FILES = {key: ARTIFACTS / name for key, name in ARTIFACT_NAMES.items()}


def _resolve_file(key: str) -> str | None:
    path = resolve_file(key, _ARTIFACT_SETTINGS)
    return None if path is None else str(path)


def _load(key: str) -> pd.DataFrame | None:
    raw = _resolve_file(key)
    if raw is None:
        return None
    return pd.read_csv(raw)


metrics = load_team_metrics()
_current_year = datetime.date.today().year
all_years = years_from_frame(metrics)
_slider_max = compute_slider_max(all_years, _current_year)
_slider_lo = all_years[0] if all_years else _current_year
all_teams = teams_from_frame(metrics)
_status = artifact_status({
    key: Path(raw) if (raw := _resolve_file(key)) else None
    for key in _FILES
})
_source_label = artifact_source_label(_ARTIFACT_SETTINGS)


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
    f"<strong>Source</strong> {html.escape(_source_label)}",
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
    _page_header("Overview")
    st.caption(
        "Team WAR is Baseball-Reference rWAR rolled up from players (`war_source=real`); "
        "Lahman wOBA/FIP approx is the fallback."
    )
    if metrics is None:
        _empty("metrics")
        return

    col_nav, col_lg = st.columns([3, 1])
    with col_nav:
        year = _season_picker("snap")
    with col_lg:
        lg = st.selectbox("League", ["All", "AL", "NL"], key="snap_lg")
    if year is None:
        _empty("season")
        return


def _page_header(label: str) -> None:
    ui.page_header(label)


def _empty(kind: str) -> None:
    ui.empty_state(kind)


def _salary_note(year: int | None) -> None:
    ui.salary_note(year)


def _season_picker(key: str = "season", default_latest: bool = True) -> int | None:
    return ui.season_picker(key, default_latest=default_latest)


def page_league_snapshot() -> None:
    overview_view.page_league_snapshot()


def page_player_explorer() -> None:
    roster_view.page_player_explorer()


def page_team_deep_dive() -> None:
    deep_dive_view.page_team_deep_dive()


def page_compare_teams() -> None:
    compare_view.page_compare_teams()


def page_contract_analysis() -> None:
    contracts_view.page_contract_analysis()


def page_efficiency_frontier() -> None:
    frontier_view.page_efficiency_frontier()


def page_whatif() -> None:
    whatif_view.page_whatif()


def page_model_insights() -> None:
    models_view.page_model_insights()


page_overview = page_league_snapshot
page_roster_lab = page_player_explorer
page_contract_watch = page_contract_analysis
page_team_profile = page_team_deep_dive
page_season_compare = page_compare_teams

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

page = ui.render_sidebar(all_years=all_years, status=_status, source=_source_label)
# nav_labels() is the product section list; keep it imported for tests/helpers.
assert page in nav_labels() or page in _PAGES
_PAGES[page]()
