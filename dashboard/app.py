"""
MLB Team Efficiency Engine — Dashboard

Pages
-----
1  League Snapshot    Full team table for any season + scatter
2  Player Explorer    All player stats, searchable, filterable by season/team/position
3  Team Profile       Single team: KPIs, season history table, roster by year
4  Season Compare     Multi-team side-by-side table + trend chart
5  Contract Analysis  All contracts tabbed by label; full searchable table
6  Efficiency Frontier  Above/below curve table + scatter
7  Standings & Phases  Window phase table + trajectory
8  What-If Sim        Payroll impact projection
9  Model Insights     Model metrics, feature importance, prediction table
"""
from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLB Efficiency Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ── Column config helpers ───────────────────────────────────────────────────────

def _money_col(label: str, unit: str = "$") -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format=f"{unit}%.1f")


def _payroll_col(label: str = "Payroll") -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="$%.1fM")


def _pct_col(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f%%")


def _dec_col(label: str, decimals: int = 2) -> st.column_config.NumberColumn:
    fmt = f"%.{decimals}f"
    return st.column_config.NumberColumn(label, format=fmt)


# Shared column config for team tables
_TEAM_COL_CFG = {
    "team_name":         st.column_config.TextColumn("Team"),
    "year_id":           st.column_config.NumberColumn("Year", format="%d"),
    "wins":              st.column_config.NumberColumn("W", format="%d"),
    "losses":            st.column_config.NumberColumn("L", format="%d"),
    "run_diff":          st.column_config.NumberColumn("Run Diff", format="%+d"),
    "payroll":           st.column_config.NumberColumn("Payroll ($M)", format="$%.1fM"),
    "payroll_per_win":   st.column_config.NumberColumn("$/Win ($M)", format="$%.2fM"),
    "wins_per_10m":      st.column_config.NumberColumn("W/$10M", format="%.2f"),
    "team_total_war":    st.column_config.NumberColumn("Team WAR", format="%.1f"),
    "cost_per_war":      st.column_config.NumberColumn("$/WAR ($M)", format="$%.2fM"),
    "war_per_1m":        st.column_config.NumberColumn("WAR/$1M", format="%.2f"),
    "surplus_value":     st.column_config.NumberColumn("Surplus ($M)", format="$%.1fM"),
    "pythag_wins":       st.column_config.NumberColumn("Pythag W", format="%.1f"),
    "pythag_gap":        st.column_config.NumberColumn("Pythag Gap", format="%+.1f"),
    "gini_salary":       st.column_config.NumberColumn("Gini", format="%.3f"),
    "dead_money_share":  st.column_config.NumberColumn("Dead Money %", format="%.1f%%"),
    "window_phase":      st.column_config.TextColumn("Phase"),
    "league_id":         st.column_config.TextColumn("Lg"),
}

# Shared column config for player tables
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
    "salary":          st.column_config.NumberColumn("Salary ($M)", format="$%.2fM"),
    "surplus_value":   st.column_config.NumberColumn("Surplus ($M)", format="$%.2fM"),
    "contract_label":  st.column_config.TextColumn("Contract"),
}


def _scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw $ to $M for display."""
    df = df.copy()
    for col in ["payroll", "max_salary", "median_salary", "payroll_per_win", "cost_per_war", "surplus_value"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    for col in ["salary"]:
        if col in df.columns:
            df[col] = df[col] / 1_000_000
    if "dead_money_share" in df.columns:
        df["dead_money_share"] = df["dead_money_share"] * 100
    return df


def _show_table(df: pd.DataFrame, col_cfg: dict | None = None, height: int = 600, **kwargs) -> None:
    cfg = {k: v for k, v in (col_cfg or {}).items() if k in df.columns}
    st.dataframe(df, column_config=cfg, use_container_width=True, height=height, **kwargs)


# ── Global state ───────────────────────────────────────────────────────────────
metrics = _load("metrics")
if metrics is None:
    st.error(
        "No artifacts found. Run the full pipeline first:\n\n"
        "```\npython3 -m pipeline.extract.pull_sources\n"
        "python3 -m pipeline.transform.build_warehouse\n"
        "python3 -m pipeline.transform.build_metrics\n"
        "python3 -m models.train_win_model\n"
        "python3 -m models.cluster_teams\n```"
    )
    st.stop()

_current_year = datetime.date.today().year
all_years = sorted(metrics["year_id"].dropna().astype(int).unique().tolist())
_slider_max = max(all_years[-1], _current_year)
all_teams = sorted(metrics["team_name"].dropna().unique().tolist())


# ── Season nav widget (reused across pages) ────────────────────────────────────
def _season_picker(key: str = "season", default_latest: bool = True) -> int:
    """Compact season selector: selectbox + ◀ ▶ buttons on one row."""
    default_idx = len(all_years) - 1 if default_latest else 0
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.button("◀", key=f"{key}_prev"):
            st.session_state[f"{key}_idx"] = max(0, st.session_state.get(f"{key}_idx", default_idx) - 1)
    with c3:
        if st.button("▶", key=f"{key}_next"):
            st.session_state[f"{key}_idx"] = min(len(all_years) - 1, st.session_state.get(f"{key}_idx", default_idx) + 1)
    idx = st.session_state.get(f"{key}_idx", default_idx)
    with c2:
        chosen = st.selectbox("Season", all_years, index=idx, key=f"{key}_sel", label_visibility="collapsed")
        st.session_state[f"{key}_idx"] = all_years.index(chosen)
    return int(chosen)


# ── Sidebar nav ────────────────────────────────────────────────────────────────
PAGES = [
    "🏟  League Snapshot",
    "👤  Player Explorer",
    "📋  Team Profile",
    "⚖️  Season Compare",
    "💰  Contract Analysis",
    "📈  Efficiency Frontier",
    "🔭  Standings & Phases",
    "🎲  What-If Sim",
    "🤖  Model Insights",
]

st.sidebar.title("⚾ MLB Efficiency Engine")
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")


# ══════════════════════════════════════════════════════════════════════════════
# 1. LEAGUE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
def page_league_snapshot() -> None:
    st.title("League Snapshot")
    st.caption("Full sortable team table for any season. Click any column header to sort.")

    col_nav, col_lg = st.columns([3, 1])
    with col_nav:
        year = _season_picker("snap")
    with col_lg:
        lg = st.selectbox("League", ["All", "AL", "NL"], key="snap_lg")

    season = metrics[metrics["year_id"] == year].copy()
    if lg != "All" and "league_id" in season.columns:
        season = season[season["league_id"] == lg]

    # KPI row
    if not season.empty:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Teams", len(season))
        k2.metric("Avg Payroll", f"${season['payroll'].mean() / 1e6:.0f}M" if season["payroll"].notna().any() else "—")
        k3.metric("Avg Wins", f"{season['wins'].mean():.0f}")
        k4.metric("Total Run Diff", f"{season['run_diff'].sum():+d}" if "run_diff" in season.columns else "—")
        if "team_total_war" in season.columns and season["team_total_war"].notna().any():
            k5.metric("Avg Team WAR", f"{season['team_total_war'].mean():.1f}")
        else:
            k5.metric("Avg Wins/$10M", f"{season['wins_per_10m'].mean():.2f}" if "wins_per_10m" in season.columns else "—")
        if "gini_salary" in season.columns and season["gini_salary"].notna().any():
            k6.metric("Avg Salary Gini", f"{season['gini_salary'].mean():.3f}")
        else:
            k6.metric("", "")

    st.divider()

    # Full table
    table_cols = [
        "team_name", "league_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ]
    table_cols = [c for c in table_cols if c in season.columns]
    display = _scale_payroll(season[table_cols]).sort_values("wins", ascending=False).reset_index(drop=True)
    _show_table(display, _TEAM_COL_CFG, height=650)

    # Chart (collapsible)
    with st.expander("Payroll vs Wins scatter", expanded=False):
        if season["payroll"].notna().any():
            fig = px.scatter(
                season.dropna(subset=["payroll", "wins"]),
                x="payroll", y="wins",
                color="window_phase" if "window_phase" in season.columns else "league_id",
                size="run_diff" if "run_diff" in season.columns else None,
                size_max=30,
                hover_name="team_name",
                labels={"payroll": "Payroll ($)", "wins": "Wins"},
                title=f"{year} — Payroll vs Wins",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLAYER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_player_explorer() -> None:
    st.title("Player Explorer")
    st.caption("All player stats for any season. Filter, search, and sort.")

    players = _load("players")
    sr_players = _load("sr_players")

    if players is None:
        st.warning("Run `python3 -m pipeline.transform.build_metrics` to generate player data.")
        return

    # ── Filters row ────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 2])
    with f1:
        yr_opts = sorted(players["year_id"].dropna().astype(int).unique().tolist())
        year = st.selectbox("Season", yr_opts, index=len(yr_opts) - 1, key="pe_year")
    with f2:
        team_opts = ["All Teams"] + sorted(players["team_name"].dropna().unique().tolist()) if "team_name" in players.columns else ["All Teams"]
        team = st.selectbox("Team", team_opts, key="pe_team")
    with f3:
        type_opts = ["All Types"]
        if "player_type" in players.columns:
            type_opts += sorted(players["player_type"].dropna().unique().tolist())
        ptype = st.selectbox("Type", type_opts, key="pe_type")
    with f4:
        name_search = st.text_input("Search player name", key="pe_name", placeholder="e.g. Judge")
    with f5:
        sort_col_opts = ["player_war", "salary", "surplus_value", "batting_war", "pitching_war", "pa", "hr", "ip", "era", "fip", "woba"]
        sort_col_opts = [c for c in sort_col_opts if c in players.columns]
        sort_by = st.selectbox("Sort by", sort_col_opts, key="pe_sort")

    # ── Apply filters ──────────────────────────────────────────────────────
    filt = players[players["year_id"] == year].copy()
    if team != "All Teams" and "team_name" in filt.columns:
        filt = filt[filt["team_name"] == team]
    if ptype != "All Types" and "player_type" in filt.columns:
        filt = filt[filt["player_type"] == ptype]
    if name_search and "name_full" in filt.columns:
        filt = filt[filt["name_full"].str.contains(name_search, case=False, na=False)]

    filt = filt.sort_values(sort_by, ascending=(sort_by in ["era", "fip"]), na_position="last").reset_index(drop=True)

    st.caption(f"{len(filt):,} players shown")

    # ── Tabs: Batting | Pitching | Contract | All ──────────────────────────
    tab_bat, tab_pit, tab_contract, tab_all = st.tabs(["Batting", "Pitching", "Contract", "All Stats"])

    bat_cols = ["name_full", "team_name", "player_type", "pa", "hr", "bb", "woba", "batting_war"]
    pit_cols = ["name_full", "team_name", "player_type", "ip", "era", "fip", "pitching_war"]
    contract_cols = ["name_full", "team_name", "player_type", "player_war", "salary", "surplus_value", "contract_label"]
    all_cols = [c for c in [
        "name_full", "team_name", "player_type",
        "pa", "hr", "bb", "woba", "batting_war",
        "ip", "era", "fip", "pitching_war",
        "player_war", "salary", "surplus_value", "contract_label",
    ] if c in filt.columns]

    with tab_bat:
        cols = [c for c in bat_cols if c in filt.columns]
        _show_table(_scale_payroll(filt[cols]), _PLAYER_COL_CFG)

    with tab_pit:
        cols = [c for c in pit_cols if c in filt.columns]
        _pit = filt[filt["ip"].notna() & (filt["ip"] > 0)] if "ip" in filt.columns else filt
        _show_table(_scale_payroll(_pit[cols]), _PLAYER_COL_CFG)

    with tab_contract:
        cols = [c for c in contract_cols if c in filt.columns]
        _show_table(_scale_payroll(filt[cols]), _PLAYER_COL_CFG)

    with tab_all:
        _show_table(_scale_payroll(filt[all_cols]), _PLAYER_COL_CFG)

    # Sportradar enriched data if available
    if sr_players is not None and not sr_players.empty:
        st.divider()
        st.subheader("Sportradar Stats (real WAR · wRC+ · ERA-)")
        sr_yr_opts = sorted(sr_players["year_id"].dropna().astype(int).unique().tolist())
        sr_year = st.selectbox("SR Season", sr_yr_opts, index=len(sr_yr_opts) - 1, key="pe_sr_year")
        sr_filt = sr_players[sr_players["year_id"] == sr_year].copy()
        if team != "All Teams" and "team_name" in sr_filt.columns:
            sr_filt = sr_filt[sr_filt["team_name"] == team]
        if name_search and "full_name" in sr_filt.columns:
            sr_filt = sr_filt[sr_filt["full_name"].str.contains(name_search, case=False, na=False)]
        sr_filt = sr_filt.sort_values("player_war_sr", ascending=False, na_position="last").reset_index(drop=True)

        sr_display_cols = [c for c in ["full_name", "team_id", "primary_position", "pa", "hr", "woba", "wrc_plus", "war", "bwar", "fwar", "ip", "era", "era_minus", "fip", "k9", "p_war", "player_war_sr"] if c in sr_filt.columns]
        sr_col_cfg = {
            "full_name":    st.column_config.TextColumn("Player"),
            "team_id":      st.column_config.TextColumn("Team"),
            "primary_position": st.column_config.TextColumn("Pos"),
            "pa":           st.column_config.NumberColumn("PA", format="%d"),
            "hr":           st.column_config.NumberColumn("HR", format="%d"),
            "woba":         st.column_config.NumberColumn("wOBA", format="%.3f"),
            "wrc_plus":     st.column_config.NumberColumn("wRC+", format="%.0f"),
            "war":          st.column_config.NumberColumn("WAR (bat)", format="%.1f"),
            "bwar":         st.column_config.NumberColumn("bWAR", format="%.1f"),
            "fwar":         st.column_config.NumberColumn("fWAR", format="%.1f"),
            "ip":           st.column_config.NumberColumn("IP", format="%.1f"),
            "era":          st.column_config.NumberColumn("ERA", format="%.2f"),
            "era_minus":    st.column_config.NumberColumn("ERA-", format="%.1f"),
            "fip":          st.column_config.NumberColumn("FIP", format="%.2f"),
            "k9":           st.column_config.NumberColumn("K/9", format="%.1f"),
            "p_war":        st.column_config.NumberColumn("pWAR", format="%.1f"),
            "player_war_sr": st.column_config.NumberColumn("Total WAR", format="%.1f"),
        }
        st.caption(f"{len(sr_filt):,} players")
        _show_table(sr_filt[sr_display_cols], sr_col_cfg)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TEAM PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def page_team_profile() -> None:
    st.title("Team Profile")

    c1, c2 = st.columns([3, 3])
    with c1:
        team = st.selectbox("Team", all_teams, key="tp_team")
    with c2:
        year = _season_picker("tp")

    team_history = metrics[metrics["team_name"] == team].sort_values("year_id")
    season_row = team_history[team_history["year_id"] == year]

    if team_history.empty:
        st.warning("No data for this team.")
        return

    # KPI row for selected season
    st.subheader(f"{team} — {year}")
    if not season_row.empty:
        r = season_row.iloc[0]
        k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
        k1.metric("Wins", int(r["wins"]) if pd.notna(r.get("wins")) else "—")
        k2.metric("Losses", int(r["losses"]) if pd.notna(r.get("losses")) else "—")
        k3.metric("Run Diff", f"{int(r['run_diff']):+d}" if pd.notna(r.get("run_diff")) else "—")
        k4.metric("Payroll", f"${r['payroll'] / 1e6:.0f}M" if pd.notna(r.get("payroll")) else "—")
        k5.metric("Team WAR", f"{r['team_total_war']:.1f}" if pd.notna(r.get("team_total_war")) else "—")
        k6.metric("W/$10M", f"{r['wins_per_10m']:.2f}" if pd.notna(r.get("wins_per_10m")) else "—")
        k7.metric("Phase", str(r.get("window_phase", "—")).title())
    else:
        st.info(f"No data for {team} in {year}.")

    st.divider()

    # Season history table
    st.subheader("Season History")
    hist_cols = [
        "year_id", "wins", "losses", "run_diff", "pythag_wins", "pythag_gap",
        "payroll", "payroll_per_win", "wins_per_10m",
        "team_total_war", "cost_per_war", "surplus_value",
        "gini_salary", "dead_money_share", "window_phase",
    ]
    hist_cols = [c for c in hist_cols if c in team_history.columns]
    hist_display = _scale_payroll(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True)
    _show_table(hist_display, _TEAM_COL_CFG, height=400)

    # Trend charts
    with st.expander("Trend charts", expanded=True):
        ch1, ch2 = st.columns(2)
        with ch1:
            fig_w = px.line(team_history, x="year_id", y="wins", markers=True, title="Wins")
            if "pythag_wins" in team_history.columns:
                fig_w.add_scatter(x=team_history["year_id"], y=team_history["pythag_wins"],
                                  mode="lines", name="Pythag W", line=dict(dash="dash", color="gray"))
            fig_w.update_layout(height=280, margin=dict(t=40, b=20))
            st.plotly_chart(fig_w, use_container_width=True)
        with ch2:
            if team_history["payroll"].notna().any():
                fig_p = px.bar(team_history, x="year_id", y="payroll", title="Payroll ($M)",
                               color_discrete_sequence=["#2a7ae2"])
                fig_p.update_yaxes(tickprefix="$", ticksuffix="M", tickformat=".0f",
                                   labelalias={"payroll": "Payroll ($M)"})
                fig_p.update_traces(customdata=team_history[["payroll"]].values / 1e6,
                                    hovertemplate="Year: %{x}<br>Payroll: $%{customdata[0]:.1f}M")
                fig_p.update_layout(height=280, margin=dict(t=40, b=20))
                st.plotly_chart(fig_p, use_container_width=True)

    # Roster for selected year
    st.subheader(f"Roster — {year}")
    players = _load("players")
    if players is not None:
        roster = players[(players["year_id"] == year)]
        if "team_name" in roster.columns:
            roster = roster[roster["team_name"] == team]
        if not roster.empty:
            roster_cols = [c for c in [
                "name_full", "player_type", "pa", "hr", "bb", "woba", "batting_war",
                "ip", "era", "fip", "pitching_war",
                "player_war", "salary", "surplus_value", "contract_label",
            ] if c in roster.columns]
            _show_table(
                _scale_payroll(roster[roster_cols]).sort_values("player_war", ascending=False).reset_index(drop=True),
                _PLAYER_COL_CFG, height=500,
            )
        else:
            st.info("No player data for this team/season.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SEASON COMPARE
# ══════════════════════════════════════════════════════════════════════════════
def page_season_compare() -> None:
    st.title("Season Compare")
    st.caption("Head-to-head team comparison across any stat and date range.")

    c1, c2 = st.columns([4, 2])
    with c1:
        selected = st.multiselect("Teams", all_teams, default=all_teams[:4], key="sc_teams")
    with c2:
        year_range = st.slider("Years", int(all_years[0]), _slider_max,
                               (max(int(all_years[0]), _slider_max - 9), _slider_max), key="sc_range")

    if len(selected) < 2:
        st.info("Select at least 2 teams.")
        return

    compare_df = metrics[
        metrics["team_name"].isin(selected) &
        metrics["year_id"].between(year_range[0], year_range[1])
    ].copy()

    if compare_df.empty:
        st.warning("No data for that combination.")
        return

    # Latest-year side-by-side table
    latest_year = int(compare_df["year_id"].max())
    st.subheader(f"Stats — {latest_year}")
    latest = compare_df[compare_df["year_id"] == latest_year]
    table_cols = [c for c in [
        "team_name", "wins", "losses", "run_diff", "pythag_wins",
        "payroll", "wins_per_10m", "team_total_war", "cost_per_war",
        "surplus_value", "gini_salary", "window_phase",
    ] if c in latest.columns]
    _show_table(_scale_payroll(latest[table_cols]).sort_values("wins", ascending=False).reset_index(drop=True),
                _TEAM_COL_CFG, height=250)

    # Full history table
    st.subheader(f"History — {year_range[0]}–{year_range[1]}")
    hist_cols = [c for c in [
        "year_id", "team_name", "wins", "run_diff", "payroll",
        "wins_per_10m", "team_total_war", "surplus_value", "window_phase",
    ] if c in compare_df.columns]
    _show_table(_scale_payroll(compare_df[hist_cols]).sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True),
                _TEAM_COL_CFG, height=400)

    # Trend chart
    metric_opts = [c for c in ["wins", "payroll", "wins_per_10m", "team_total_war",
                                "cost_per_war", "surplus_value", "run_diff", "gini_salary"]
                   if c in compare_df.columns]
    y_metric = st.selectbox("Chart metric", metric_opts, key="sc_metric")

    plot_df = compare_df.copy()
    if y_metric == "payroll":
        plot_df["payroll"] = plot_df["payroll"] / 1e6
        y_label = "Payroll ($M)"
    else:
        y_label = y_metric

    fig = px.line(plot_df, x="year_id", y=y_metric, color="team_name",
                  markers=True, title=f"{y_label} — {year_range[0]}–{year_range[1]}",
                  labels={y_metric: y_label})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONTRACT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_contract_analysis() -> None:
    st.title("Contract Analysis")
    st.caption("Every player contract, classified and searchable. Salary data from Lahman (through 2016).")

    players = _load("players")
    if players is None:
        st.warning("No player data. Run `python3 -m pipeline.transform.build_metrics`.")
        return

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        yr_opts = sorted(players["year_id"].dropna().astype(int).unique().tolist())
        year = st.selectbox("Season", ["All Seasons"] + yr_opts, key="ca_year")
    with f2:
        team_opts = ["All Teams"] + sorted(players["team_name"].dropna().unique().tolist()) if "team_name" in players.columns else ["All Teams"]
        team = st.selectbox("Team", team_opts, key="ca_team")
    with f3:
        name_search = st.text_input("Search player", key="ca_name", placeholder="e.g. Bonds")

    filt = players.copy()
    if year != "All Seasons":
        filt = filt[filt["year_id"] == int(year)]
    if team != "All Teams" and "team_name" in filt.columns:
        filt = filt[filt["team_name"] == team]
    if name_search and "name_full" in filt.columns:
        filt = filt[filt["name_full"].str.contains(name_search, case=False, na=False)]
    filt = filt[filt["salary"] > 0] if "salary" in filt.columns else filt

    contract_cols = [c for c in [
        "name_full", "year_id", "team_name", "player_type",
        "player_war", "salary", "surplus_value", "contract_label",
        "batting_war", "pitching_war", "pa", "ip",
    ] if c in filt.columns]

    tabs = st.tabs(["All Contracts", "Surplus Value", "Overpaid", "Dead Money", "Fair Value"])

    def _contract_table(df: pd.DataFrame, sort: str, asc: bool = False) -> None:
        display = _scale_payroll(df[contract_cols]).sort_values(sort, ascending=asc, na_position="last").reset_index(drop=True)
        st.caption(f"{len(display):,} contracts")
        _show_table(display, _PLAYER_COL_CFG)

    with tabs[0]:
        _contract_table(filt, "surplus_value", asc=False)

    with tabs[1]:
        sv = filt[filt.get("contract_label", pd.Series()) == "surplus_value"] if "contract_label" in filt.columns else filt[filt["surplus_value"] > 2e6]
        _contract_table(sv, "surplus_value", asc=False)

    with tabs[2]:
        op = filt[filt.get("contract_label", pd.Series()) == "overpaid"] if "contract_label" in filt.columns else filt
        _contract_table(op, "surplus_value", asc=True)

    with tabs[3]:
        dm = filt[filt.get("contract_label", pd.Series()) == "dead_money"] if "contract_label" in filt.columns else filt
        _contract_table(dm, "salary", asc=False)

    with tabs[4]:
        fv = filt[filt.get("contract_label", pd.Series()) == "fair_value"] if "contract_label" in filt.columns else filt
        _contract_table(fv, "player_war", asc=False)

    # Scatter
    with st.expander("WAR vs Salary scatter", expanded=False):
        if "salary" in filt.columns and "player_war" in filt.columns:
            plot_f = _scale_payroll(filt.dropna(subset=["salary", "player_war"]))
            fig = px.scatter(
                plot_f, x="salary", y="player_war",
                color="contract_label" if "contract_label" in plot_f.columns else None,
                hover_name="name_full" if "name_full" in plot_f.columns else None,
                hover_data=["year_id", "team_name"] if "team_name" in plot_f.columns else [],
                labels={"salary": "Salary ($M)", "player_war": "WAR"},
                color_discrete_map={"surplus_value": "#2ca02c", "fair_value": "#1f77b4",
                                    "overpaid": "#ff7f0e", "dead_money": "#d62728"},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6. EFFICIENCY FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
def page_efficiency_frontier() -> None:
    st.title("Efficiency Frontier")
    st.caption("Teams above the curve produce more wins per dollar than the league baseline.")

    frontier_data = _load("frontier_data")
    clusters = _load("clusters")

    frontier_tab, cluster_tab = st.tabs(["Frontier", "Team Archetypes"])

    with frontier_tab:
        if frontier_data is None:
            st.info("Run `python3 -m models.train_win_model` to generate frontier data.")
        else:
            fd = frontier_data.copy()
            fd["above_label"] = fd["above_frontier"].map({True: "Above (Efficient)", False: "Below (Wasteful)"})

            yr_range = st.slider("Years", int(fd["year_id"].min()),
                                 max(int(fd["year_id"].max()), _current_year),
                                 (int(fd["year_id"].min()), max(int(fd["year_id"].max()), _current_year)),
                                 key="ef_range")
            fd = fd[fd["year_id"].between(yr_range[0], yr_range[1])]

            # Table first
            table_cols = [c for c in ["year_id", "team_name", "payroll_m", "wins", "frontier_pred", "above_frontier", "above_label"] if c in fd.columns]
            fd_display = fd[table_cols].sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True)
            ef_col_cfg = {
                "year_id":       st.column_config.NumberColumn("Year", format="%d"),
                "team_name":     st.column_config.TextColumn("Team"),
                "payroll_m":     st.column_config.NumberColumn("Payroll ($M)", format="$%.1fM"),
                "wins":          st.column_config.NumberColumn("Wins", format="%d"),
                "frontier_pred": st.column_config.NumberColumn("Frontier Pred", format="%.1f"),
                "above_frontier": st.column_config.CheckboxColumn("Above Curve"),
                "above_label":   st.column_config.TextColumn("Status"),
            }
            n_above = int(fd["above_frontier"].sum())
            st.caption(f"{n_above:,} of {len(fd):,} team-seasons above the efficiency frontier")
            _show_table(fd_display, ef_col_cfg, height=500)

            with st.expander("Scatter", expanded=False):
                fig = px.scatter(fd, x="payroll_m", y="wins", color="above_label",
                                 hover_name="team_name", hover_data=["year_id"],
                                 labels={"payroll_m": "Payroll ($M)", "wins": "Wins"},
                                 color_discrete_map={"Above (Efficient)": "#2ca02c", "Below (Wasteful)": "#d62728"})
                if "frontier_pred" in fd.columns:
                    fl = fd.sort_values("payroll_m")[["payroll_m", "frontier_pred"]].drop_duplicates()
                    fig.add_trace(go.Scatter(x=fl["payroll_m"], y=fl["frontier_pred"],
                                            mode="lines", line=dict(color="#1f77b4", dash="dash", width=2),
                                            name="Frontier"))
                fig.update_layout(height=480)
                st.plotly_chart(fig, use_container_width=True)

    with cluster_tab:
        if clusters is None:
            st.info("Run `python3 -m models.cluster_teams` to generate cluster data.")
        else:
            cluster_summ = _load("cluster_summ")
            if cluster_summ is not None:
                st.subheader("Archetype Summary")
                _show_table(cluster_summ, height=250)

            st.subheader("All Team-Season Cluster Assignments")
            clust_cols = [c for c in ["year_id", "team_name", "cluster_label", "wins", "payroll",
                                      "team_total_war", "wins_per_10m", "window_phase"] if c in clusters.columns]
            clust_display = _scale_payroll(clusters[clust_cols]).sort_values(["year_id", "wins"], ascending=[False, False]).reset_index(drop=True)
            clust_cfg = {**_TEAM_COL_CFG, "cluster_label": st.column_config.TextColumn("Archetype")}
            _show_table(clust_display, clust_cfg, height=500)

            with st.expander("Scatter by archetype", expanded=False):
                fig = px.scatter(clusters.dropna(subset=["payroll", "wins"]),
                                 x="payroll", y="wins", color="cluster_label",
                                 hover_name="team_name", hover_data=["year_id"])
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. STANDINGS & PHASES
# ══════════════════════════════════════════════════════════════════════════════
def page_standings_phases() -> None:
    st.title("Standings & Phases")
    st.caption("Current window phase per franchise and full historical standings.")

    window_df = _load("window")
    phase_tab, standings_tab = st.tabs(["Window Phases", "Historical Standings"])

    with phase_tab:
        if window_df is None:
            st.info("No window phase data. Run `python3 -m pipeline.transform.build_metrics`.")
        else:
            phase_col_cfg = {
                "team_name":    st.column_config.TextColumn("Team"),
                "year_id":      st.column_config.NumberColumn("Latest Year", format="%d"),
                "window_phase": st.column_config.TextColumn("Phase"),
                "wins":         st.column_config.NumberColumn("Wins", format="%d"),
                "payroll":      st.column_config.NumberColumn("Payroll ($M)", format="$%.0fM"),
                "team_total_war": st.column_config.NumberColumn("WAR", format="%.1f"),
            }
            display = window_df.copy()
            if "payroll" in display.columns:
                display["payroll"] = display["payroll"] / 1e6
            phases = ["All"] + sorted(display["window_phase"].dropna().unique().tolist()) if "window_phase" in display.columns else ["All"]
            phase_filter = st.selectbox("Filter by phase", phases, key="sp_phase")
            if phase_filter != "All":
                display = display[display["window_phase"] == phase_filter]
            _show_table(display.sort_values("wins", ascending=False).reset_index(drop=True), phase_col_cfg, height=500)

    with standings_tab:
        year = _season_picker("sp_year")
        season = metrics[metrics["year_id"] == year].copy()
        if season.empty:
            st.info(f"No data for {year}.")
        else:
            for lg in ["AL", "NL"]:
                if "league_id" not in season.columns:
                    break
                lg_df = season[season["league_id"] == lg]
                if lg_df.empty:
                    continue
                st.subheader(lg)
                cols = [c for c in ["team_name", "wins", "losses", "run_diff", "payroll", "wins_per_10m", "team_total_war", "window_phase"] if c in lg_df.columns]
                _show_table(_scale_payroll(lg_df[cols]).sort_values("wins", ascending=False).reset_index(drop=True),
                            _TEAM_COL_CFG, height=320)


# ══════════════════════════════════════════════════════════════════════════════
# 8. WHAT-IF SIM
# ══════════════════════════════════════════════════════════════════════════════
def page_whatif() -> None:
    st.title("What-If Simulation")
    st.caption("Estimate win change from a payroll increase using league-wide regression.")

    c1, c2 = st.columns(2)
    with c1:
        team = st.selectbox("Team", all_teams, key="wi_team")
    with c2:
        year = _season_picker("wi")

    team_history = metrics[metrics["team_name"] == team].sort_values("year_id")
    row = team_history[team_history["year_id"] == year]
    if row.empty:
        row = team_history.iloc[[-1]]
        st.info(f"No data for {year} — showing {int(row.iloc[0]['year_id'])}")

    r = row.iloc[0]
    current_payroll = float(r.get("payroll", 0) or 0)
    current_wins = float(r.get("wins", 0) or 0)
    current_war = r.get("team_total_war")

    st.subheader(f"{team} — {int(r['year_id'])} Baseline")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Payroll", f"${current_payroll / 1e6:.0f}M")
    k2.metric("Wins", int(current_wins))
    k3.metric("W/$10M", f"{r['wins_per_10m']:.2f}" if pd.notna(r.get("wins_per_10m")) else "—")
    if pd.notna(current_war):
        k4.metric("Team WAR", f"{current_war:.1f}")

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
    k1.metric("New Payroll", f"${new_payroll / 1e6:.0f}M", delta=f"{payroll_delta_m:+.0f}M")
    k2.metric("Projected Wins", f"{projected_wins:.0f}", delta=f"{win_gain:+.1f}")
    k3.metric("New $/Win", f"${new_payroll / max(projected_wins, 1) / 1e6:.2f}M" if projected_wins > 0 else "—")

    st.caption("Linear regression on all historical team-seasons. Actual results depend on how payroll is allocated.")

    # History table
    st.subheader("Historical Record")
    hist_cols = [c for c in ["year_id", "wins", "run_diff", "payroll", "wins_per_10m", "team_total_war", "window_phase"] if c in team_history.columns]
    _show_table(_scale_payroll(team_history[hist_cols]).sort_values("year_id", ascending=False).reset_index(drop=True),
                _TEAM_COL_CFG, height=350)

    fig = px.line(team_history, x="year_id", y="wins", markers=True, title=f"{team} — Win History")
    fig.add_scatter(x=[int(r["year_id"]) + 1], y=[projected_wins],
                    mode="markers+text", marker=dict(color="orange", size=14, symbol="star"),
                    text=["Projected"], textposition="top center", name="Projection")
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9. MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
def page_model_insights() -> None:
    st.title("Model Insights")

    model_metrics_df = _load("model_metrics")
    importance = _load("importance")
    preds = _load("preds")

    perf_tab, feat_tab, pred_tab = st.tabs(["Performance", "Feature Importance", "Predictions"])

    with perf_tab:
        if model_metrics_df is not None:
            cfg = {
                "model": st.column_config.TextColumn("Model"),
                "mae":   st.column_config.NumberColumn("MAE (wins)", format="%.2f"),
                "r2":    st.column_config.NumberColumn("R²", format="%.4f"),
                "n_rows": st.column_config.NumberColumn("N", format="%d"),
            }
            _show_table(model_metrics_df, cfg, height=150)

    with feat_tab:
        if importance is not None:
            cfg = {
                "feature":    st.column_config.TextColumn("Feature"),
                "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
            }
            _show_table(importance.sort_values("importance", ascending=False).reset_index(drop=True), cfg, height=450)
            fig = px.bar(importance.head(15), x="importance", y="feature", orientation="h",
                         color="importance", color_continuous_scale="Blues",
                         title="Top 15 Features")
            fig.update_layout(height=420, yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)

    with pred_tab:
        if preds is not None:
            err_col = "absolute_error_xgb" if "absolute_error_xgb" in preds.columns else \
                      "absolute_error_lr" if "absolute_error_lr" in preds.columns else None
            pred_cfg = {
                "team_name":            st.column_config.TextColumn("Team"),
                "year_id":              st.column_config.NumberColumn("Year", format="%d"),
                "actual_wins":          st.column_config.NumberColumn("Actual W", format="%d"),
                "predicted_wins_xgb":   st.column_config.NumberColumn("XGB Pred", format="%.1f"),
                "predicted_wins_lr":    st.column_config.NumberColumn("LR Pred", format="%.1f"),
                "absolute_error_xgb":   st.column_config.NumberColumn("XGB Error", format="%.1f"),
                "absolute_error_lr":    st.column_config.NumberColumn("LR Error", format="%.1f"),
            }
            sort_df = preds.sort_values(err_col, ascending=False).reset_index(drop=True) if err_col else preds
            st.caption(f"{len(sort_df):,} predictions — sorted by largest error")
            _show_table(sort_df, pred_cfg, height=500)

            if "actual_wins" in preds.columns and "predicted_wins_xgb" in preds.columns:
                with st.expander("Actual vs Predicted scatter", expanded=False):
                    fig = px.scatter(preds, x="actual_wins", y="predicted_wins_xgb",
                                     hover_name="team_name", hover_data=["year_id"],
                                     labels={"actual_wins": "Actual Wins", "predicted_wins_xgb": "XGB Predicted"})
                    lo = preds[["actual_wins", "predicted_wins_xgb"]].min().min() - 2
                    hi = preds[["actual_wins", "predicted_wins_xgb"]].max().max() + 2
                    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                            line=dict(dash="dash", color="red"), name="Perfect"))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)


# ── Routing ────────────────────────────────────────────────────────────────────
_PAGES = {
    "🏟  League Snapshot":   page_league_snapshot,
    "👤  Player Explorer":   page_player_explorer,
    "📋  Team Profile":      page_team_profile,
    "⚖️  Season Compare":    page_season_compare,
    "💰  Contract Analysis": page_contract_analysis,
    "📈  Efficiency Frontier": page_efficiency_frontier,
    "🔭  Standings & Phases": page_standings_phases,
    "🎲  What-If Sim":       page_whatif,
    "🤖  Model Insights":    page_model_insights,
}

_PAGES[page]()
