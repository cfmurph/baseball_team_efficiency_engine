"""Shared Streamlit chrome: theme inject, nav, headers, tables, charts."""
from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from dashboard.helpers import (
    PRIOR_SEASON_TABLE_NOTE,
    app_frame_html,
    artifact_status,
    clamp_season_for_page,
    empty_state_copy,
    masthead_html,
    nav_groups,
    nav_page,
    salary_coverage_note,
    year_span_label,
)
from dashboard.state import NAV_PAGE, SEASON_YEAR, SELECTED_LEAGUE, SELECTED_TEAM
from dashboard.theme import APP_CSS, PLOTLY_CONFIG, PLOTLY_LAYOUT, SCATTER_MARKER, TOKENS
from src.baseball_analytics.dashboard_utils import apply_plotly_layout, scale_payroll_for_display


def inject_theme() -> None:
    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


def page_header(label: str, extra_caption: str | None = None) -> None:
    st.markdown(masthead_html(label, extra_caption), unsafe_allow_html=True)


def empty_state(kind: str) -> None:
    copy = empty_state_copy(kind)
    cmd = copy.get("command") or ""
    cmd_html = f"<pre><code>{html.escape(cmd)}</code></pre>" if cmd else ""
    st.markdown(
        f"""
        <div class="empty-card" role="status">
          <h3>{html.escape(copy["title"])}</h3>
          <p>{html.escape(copy["body"])}</p>
          {cmd_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def salary_note(year: int | None) -> None:
    note = salary_coverage_note(year)
    if note:
        st.info(note)


def prior_season_note(*, show: bool, message: str | None = None) -> None:
    if show:
        st.info(message or PRIOR_SEASON_TABLE_NOTE)


def panel_head(title: str, hint: str = "") -> None:
    hint_html = f'<span class="hint">{html.escape(hint)}</span>' if hint else ""
    st.markdown(
        f'<div class="panel-head"><span class="title">{html.escape(title)}</span>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def show_table(df: pd.DataFrame, col_cfg: dict | None = None, height: int = 600, **kwargs) -> None:
    cfg = {k: v for k, v in (col_cfg or {}).items() if k in df.columns}
    kwargs.setdefault("hide_index", True)
    st.dataframe(df, column_config=cfg, use_container_width=True, height=height, **kwargs)


def apply_chart_layout(fig: Any, layout: dict | None = None) -> None:
    apply_plotly_layout(fig, layout or PLOTLY_LAYOUT)


def chart(fig: Any, height: int = 400) -> None:
    apply_chart_layout(fig)
    fig.update_layout(height=height, title_text="")
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def scale_payroll(df: pd.DataFrame) -> pd.DataFrame:
    return scale_payroll_for_display(df)


ALL_YEARS: list[int] = []


def season_picker(key: str = SEASON_YEAR, default_latest: bool = True, years: list[int] | None = None) -> int | None:
    """Season control for team pages. Does not clobber a player-only ``season_year``."""
    year_opts = years if years is not None else ALL_YEARS
    if not year_opts:
        return None
    current = st.session_state.get(SEASON_YEAR)
    display, write_shared = clamp_season_for_page(
        current if current is None else int(current),
        year_opts,
        default_latest=default_latest,
    )
    widget_key = key if key != SEASON_YEAR else f"{SEASON_YEAR}_team"
    if current not in year_opts or st.session_state.get(widget_key) not in year_opts:
        st.session_state[widget_key] = display
    elif write_shared and st.session_state.get(widget_key) != current:
        st.session_state[widget_key] = current
    c1, c2, c3 = st.columns([1, 6, 1])
    with c1:
        if st.button("◀", key=f"{widget_key}_prev", help="Previous season"):
            idx = year_opts.index(st.session_state[widget_key])
            st.session_state[widget_key] = year_opts[max(0, idx - 1)]
    with c3:
        if st.button("▶", key=f"{widget_key}_next", help="Next season"):
            idx = year_opts.index(st.session_state[widget_key])
            st.session_state[widget_key] = year_opts[min(len(year_opts) - 1, idx + 1)]
    with c2:
        st.selectbox("Season", year_opts, key=widget_key, label_visibility="collapsed")
    chosen = int(st.session_state[widget_key])
    if write_shared or chosen != display:
        st.session_state[SEASON_YEAR] = chosen
    return chosen


def team_select(all_teams: list[str], *, label: str = "Team") -> str | None:
    """Single-franchise control bound to shared ``selected_team``."""
    if not all_teams:
        return None
    current = st.session_state.get(SELECTED_TEAM)
    if current not in all_teams:
        st.session_state[SELECTED_TEAM] = all_teams[0]
    return str(st.selectbox(label, all_teams, key=SELECTED_TEAM))


def league_select() -> str:
    """League filter bound to shared ``selected_league``."""
    options = ["All", "AL", "NL"]
    if st.session_state.get(SELECTED_LEAGUE) not in options:
        st.session_state[SELECTED_LEAGUE] = "All"
    return str(st.selectbox("League", options, key=SELECTED_LEAGUE))


def render_app_frame(*, all_years: list[int], status: dict, source: str, page: str) -> None:
    """Top command strip: seasons, artifacts, source, active desk."""
    st.markdown(
        app_frame_html(
            seasons=year_span_label(all_years),
            artifacts=f"{status['n_present']}/{status['n_total']}",
            source=source,
            page=page,
        ),
        unsafe_allow_html=True,
    )


def render_sidebar(*, all_years: list[int], status: dict, source: str = "local") -> str:
    """Numbered rail nav. Returns the selected page label."""
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <div class="wordmark">Efficiency<em>Engine</em></div>
          <small>Front office · MLB</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if NAV_PAGE not in st.session_state:
        st.session_state[NAV_PAGE] = "Overview"

    for group_name, pages in nav_groups():
        st.sidebar.markdown(f'<div class="nav-group">{html.escape(group_name)}</div>', unsafe_allow_html=True)
        for page in pages:
            label = page["label"]
            index = page.get("index", "")
            selected = st.session_state[NAV_PAGE] == label
            if st.sidebar.button(
                f"{index}  {label}",
                key=f"nav_{page['key']}",
                use_container_width=True,
                type="primary" if selected else "secondary",
            ):
                st.session_state[NAV_PAGE] = label

    missing_core = [key for key in ("metrics", "players", "frontier_data", "preds") if key in status["missing"]]
    ready = not missing_core
    pill = (
        '<span class="status-pill live"><span class="dot"></span>Live</span>'
        if ready
        else '<span class="status-pill setup"><span class="dot"></span>Setup needed</span>'
    )
    note = "Core files missing — pages show setup steps." if missing_core else "Payroll metrics are fullest for 1990–2016."
    st.sidebar.markdown(
        f"""
        <div class="sidebar-status">
          <div class="status-row"><span>Seasons</span><strong>{html.escape(year_span_label(all_years))}</strong></div>
          <div class="status-row"><span>Artifacts</span><strong>{status['n_present']}/{status['n_total']}</strong></div>
          <div class="status-row"><span>Source</span><strong>{html.escape(source)}</strong></div>
          <div>{html.escape(note)}</div>
          {pill}
        </div>
        """,
        unsafe_allow_html=True,
    )
    return st.session_state[NAV_PAGE]


def team_column_config() -> dict:
    return {
        "rank":              st.column_config.NumberColumn("#", format="%d", width="small"),
        "team_name":         st.column_config.TextColumn("Team", width="medium"),
        "year_id":           st.column_config.NumberColumn("Year", format="%d", width="small"),
        "wins":              st.column_config.NumberColumn("W", format="%d", width="small"),
        "losses":            st.column_config.NumberColumn("L", format="%d", width="small"),
        "run_diff":          st.column_config.NumberColumn("RD", format="%+d", width="small"),
        "payroll":           st.column_config.NumberColumn("Payroll", format="$%.1fM"),
        "payroll_per_win":   st.column_config.NumberColumn("$/W", format="$%.2fM"),
        "wins_per_10m":      st.column_config.NumberColumn("W/$10M", format="%.2f"),
        "team_total_war":    st.column_config.NumberColumn("WAR", format="%.1f"),
        "war_source":        st.column_config.TextColumn("Src", width="small"),
        "cost_per_war":      st.column_config.NumberColumn("$/WAR", format="$%.2fM"),
        "war_per_1m":        st.column_config.NumberColumn("WAR/$1M", format="%.2f"),
        "surplus_value":     st.column_config.NumberColumn("Surplus", format="$%.1fM"),
        "pythag_wins":       st.column_config.NumberColumn("Pyth W", format="%.1f"),
        "pythag_gap":        st.column_config.NumberColumn("Gap", format="%+.1f"),
        "gini_salary":       st.column_config.NumberColumn("Gini", format="%.3f"),
        "dead_money_share":  st.column_config.NumberColumn("Dead %", format="%.1f%%"),
        "window_phase":      st.column_config.TextColumn("Phase"),
        "league_id":         st.column_config.TextColumn("Lg", width="small"),
        "efficiency_label":  st.column_config.TextColumn("Eff"),
    }


def player_column_config() -> dict:
    return {
        "name_full":        st.column_config.TextColumn("Player"),
        "player_id":        st.column_config.TextColumn("ID", width="small"),
        "year_id":          st.column_config.NumberColumn("Year", format="%d"),
        "team_name":        st.column_config.TextColumn("Team"),
        "player_type":      st.column_config.TextColumn("Type", width="small"),
        "primary_position": st.column_config.TextColumn("Pos", width="small"),
        "pa":               st.column_config.NumberColumn("PA", format="%d"),
        "hr":               st.column_config.NumberColumn("HR", format="%d"),
        "bb":               st.column_config.NumberColumn("BB", format="%d"),
        "woba":             st.column_config.NumberColumn("wOBA", format="%.3f"),
        "ip":               st.column_config.NumberColumn("IP", format="%.1f"),
        "era":              st.column_config.NumberColumn("ERA", format="%.2f"),
        "fip":              st.column_config.NumberColumn("FIP", format="%.2f"),
        "batting_war":      st.column_config.NumberColumn("bWAR", format="%.1f"),
        "pitching_war":     st.column_config.NumberColumn("pWAR", format="%.1f"),
        "player_war":       st.column_config.NumberColumn("WAR", format="%.1f"),
        "war_source":       st.column_config.TextColumn("Src", width="small"),
        "salary":           st.column_config.NumberColumn("Salary", format="$%.2fM"),
        "surplus_value":    st.column_config.NumberColumn("Surplus", format="$%.2fM"),
        "contract_label":   st.column_config.TextColumn("Contract"),
    }


# Re-export marker so views can import from one place
__all__ = [
    "SCATTER_MARKER",
    "TOKENS",
    "artifact_status",
    "apply_chart_layout",
    "chart",
    "empty_state",
    "inject_theme",
    "page_header",
    "panel_head",
    "player_column_config",
    "prior_season_note",
    "render_app_frame",
    "render_sidebar",
    "salary_note",
    "scale_payroll",
    "season_picker",
    "show_table",
    "team_column_config",
]
