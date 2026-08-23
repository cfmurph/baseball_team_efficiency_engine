"""Design tokens + CSS + Plotly theme for the front-office dashboard.

Scoreboard / ops-console system: condensed display type, mono stats,
hairline rules, crimson accent. Pages should import tokens from here
instead of hard-coding colors.
"""
from __future__ import annotations

# ── Tokens ───────────────────────────────────────────────────────────────────
BG = "#03040a"
BG_ELEVATED = "#07080f"
SURFACE = "#0c1018"
SURFACE_2 = "#121826"
BORDER = "#1e2836"
BORDER_SUBTLE = "#151c28"
TEXT = "#f4f1ea"
TEXT_MUTED = "#9aa4b2"
TEXT_DIM = "#5d6876"
CRIMSON = "#ff2d3a"
CRIMSON_DEEP = "#c8102e"
CRIMSON_SOFT = "rgba(255, 45, 58, 0.16)"
CYAN = "#6ecbff"
GREEN = "#3ee08f"
AMBER = "#f5c518"
PURPLE = "#b794f6"
ORANGE = "#ff7a59"

FONT_DISPLAY = '"Barlow Condensed", "Oswald", "Impact", sans-serif'
FONT_UI = '"IBM Plex Sans", "Helvetica Neue", sans-serif'
FONT_MONO = '"IBM Plex Mono", "ui-monospace", monospace'

# Shared token map — pages/charts import from here, not ad-hoc hex.
TOKENS = {
    "bg": BG,
    "bg_elevated": BG_ELEVATED,
    "surface": SURFACE,
    "surface_2": SURFACE_2,
    "border": BORDER,
    "border_subtle": BORDER_SUBTLE,
    "text": TEXT,
    "text_muted": TEXT_MUTED,
    "text_dim": TEXT_DIM,
    "crimson": CRIMSON,
    "crimson_deep": CRIMSON_DEEP,
    "cyan": CYAN,
    "green": GREEN,
    "amber": AMBER,
    "purple": PURPLE,
    "orange": ORANGE,
    "font_display": FONT_DISPLAY,
    "font_ui": FONT_UI,
    "font_mono": FONT_MONO,
}

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color=TEXT_MUTED, size=11),
    title=dict(text="", x=0.0, pad=dict(t=0, b=0)),
    xaxis=dict(
        gridcolor="#121826",
        linecolor=BORDER,
        tickcolor=BORDER,
        zeroline=False,
        tickfont=dict(color=TEXT_DIM, size=10, family="IBM Plex Mono, monospace"),
        title_font=dict(color=TEXT_DIM, size=10),
        showspikes=False,
    ),
    yaxis=dict(
        gridcolor="#121826",
        linecolor=BORDER,
        tickcolor=BORDER,
        zeroline=False,
        tickfont=dict(color=TEXT_DIM, size=10, family="IBM Plex Mono, monospace"),
        title_font=dict(color=TEXT_DIM, size=10),
        showspikes=False,
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        borderwidth=0,
        font=dict(size=10, color=TEXT_MUTED),
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor=SURFACE_2,
        bordercolor=BORDER,
        font=dict(family="IBM Plex Sans, sans-serif", size=11, color=TEXT),
        align="left",
    ),
    margin=dict(t=28, b=36, l=44, r=12),
    colorway=[CRIMSON, CYAN, GREEN, AMBER, PURPLE, ORANGE, "#60a5fa"],
)

SCATTER_MARKER = dict(size=9, opacity=0.9, line=dict(width=0.4, color=BG))

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}


def inject_theme() -> None:
    """Apply APP_CSS. Also exported from ``dashboard.ui``."""
    import streamlit as st

    st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)

APP_CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ── Kill Streamlit chrome ──────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
#MainMenu, footer, .stDeployButton,
div[data-testid="stStatusWidget"] {{
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
}}
.stApp {{
    background:
        radial-gradient(1200px 480px at 0% -10%, rgba(255,45,58,0.10), transparent 55%),
        linear-gradient(180deg, #05060c 0%, {BG} 28%);
    color: {TEXT};
    font-family: {FONT_UI};
}}
.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: 48px 48px;
}}
.stApp > div {{ position: relative; z-index: 1; }}
.stApp, .stMarkdown, p, label {{
    font-family: {FONT_UI} !important;
}}
.block-container {{
    padding: 0.35rem 1.5rem 3.2rem !important;
    max-width: 1600px;
}}
[data-testid="stVerticalBlock"] > div:has(> .app-frame) {{
    margin-bottom: 0;
}}

/* ── Top command frame ─────────────────────────────────── */
.app-frame {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0 0.65rem;
    margin: 0 0 0.85rem;
    border-bottom: 2px solid {CRIMSON};
}}
.frame-brand {{
    font-family: {FONT_DISPLAY};
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT};
    line-height: 1;
}}
.frame-brand span {{ color: {CRIMSON}; margin-left: 0.28em; }}
.frame-meta {{
    font-family: {FONT_MONO};
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}
.frame-meta b {{ color: {TEXT}; font-weight: 600; }}
.frame-meta i {{
    font-style: normal;
    width: 1px; height: 10px;
    background: {BORDER};
    display: inline-block;
}}

/* ── Sidebar rail ──────────────────────────────────────── */
section[data-testid="stSidebar"] {{
    background: {BG_ELEVATED} !important;
    border-right: 1px solid {BORDER_SUBTLE} !important;
    min-width: 232px !important;
}}
section[data-testid="stSidebar"] > div:first-child {{
    padding: 0.85rem 0.7rem 1.2rem;
}}
.sidebar-brand {{
    padding: 0.15rem 0.2rem 0.85rem;
    margin-bottom: 0.35rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.sidebar-brand .wordmark {{
    font-family: {FONT_DISPLAY};
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT};
    line-height: 0.95;
}}
.sidebar-brand .wordmark em {{
    font-style: normal;
    color: {CRIMSON};
    display: block;
}}
.sidebar-brand small {{
    display: block;
    margin-top: 0.35rem;
    font-family: {FONT_MONO};
    font-size: 0.58rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {TEXT_DIM};
}}
.nav-group {{
    font-family: {FONT_MONO};
    color: {TEXT_DIM};
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin: 0.85rem 0.15rem 0.25rem;
    padding-left: 0.15rem;
}}

[data-testid="stSidebar"] .stButton {{ margin-bottom: 1px; }}
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: transparent !important;
    color: {TEXT_MUTED} !important;
    border: 0 !important;
    border-left: 3px solid transparent !important;
    border-radius: 0 !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-family: {FONT_DISPLAY} !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase !important;
    padding: 0.38rem 0.55rem !important;
    min-height: 2rem !important;
    box-shadow: none !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,45,58,0.07) !important;
    color: {TEXT} !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {{
    background: {CRIMSON_SOFT} !important;
    color: {TEXT} !important;
    border-left: 3px solid {CRIMSON} !important;
    font-weight: 800 !important;
}}

.sidebar-status {{
    margin-top: 1.1rem;
    padding: 0.75rem 0.35rem 0;
    border-top: 1px solid {BORDER_SUBTLE};
    color: {TEXT_DIM};
    font-family: {FONT_MONO};
    font-size: 0.66rem;
    letter-spacing: 0.04em;
    line-height: 1.55;
}}
.sidebar-status .status-row {{
    display: flex; justify-content: space-between; gap: 0.4rem;
    margin-bottom: 0.28rem;
    text-transform: uppercase;
}}
.sidebar-status strong {{ color: {TEXT}; font-weight: 600; }}
.status-pill {{
    display: inline-flex; align-items: center; gap: 0.35rem;
    margin-top: 0.5rem;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.status-pill.live {{ color: {GREEN}; }}
.status-pill.setup {{ color: {AMBER}; }}
.status-pill .dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
}}

/* ── Masthead ─────────────────────────────────────────── */
.masthead {{
    margin: 0 0 0.85rem;
    padding: 0 0 0.7rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.masthead .kicker {{
    font-family: {FONT_MONO};
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {CRIMSON};
    margin-bottom: 0.1rem;
}}
.masthead h1 {{
    font-family: {FONT_DISPLAY} !important;
    font-size: 2.55rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
    color: {TEXT} !important;
    line-height: 0.92 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}}
.masthead .blurb {{
    margin: 0.4rem 0 0;
    color: {TEXT_MUTED};
    font-size: 0.86rem;
    max-width: 46rem;
    line-height: 1.45;
}}
.war-note {{
    color: {TEXT_DIM};
    font-size: 0.74rem;
    margin: 0.2rem 0 0;
}}

/* Kill leftover Streamlit titles if any page still emits them */
h1 {{
    font-family: {FONT_DISPLAY} !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.02em !important;
    border-bottom: none !important;
    color: {TEXT} !important;
}}
h2 {{
    font-family: {FONT_DISPLAY} !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {TEXT} !important;
    margin-top: 0.2rem !important;
}}
h3 {{
    font-family: {FONT_MONO} !important;
    font-size: 0.66rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    color: {TEXT_DIM} !important;
}}
.stCaption, [data-testid="stCaptionContainer"] {{
    color: {TEXT_DIM} !important;
    font-size: 0.78rem !important;
}}

.panel-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.85rem 0 0.4rem;
    padding-bottom: 0.28rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.panel-head .title {{
    font-family: {FONT_DISPLAY};
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {TEXT};
}}
.panel-head .hint {{
    font-family: {FONT_MONO};
    font-size: 0.64rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {TEXT_DIM};
}}

/* ── KPI score-ribbon ─────────────────────────────────── */
.kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0;
    margin: 0.15rem 0 1rem;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {CRIMSON};
}}
.kpi-card {{
    padding: 0.7rem 0.85rem 0.65rem;
    border-right: 1px solid {BORDER_SUBTLE};
    min-height: 78px;
    background: transparent;
    box-shadow: none;
}}
.kpi-card:last-child {{ border-right: none; }}
.kpi-label {{
    font-family: {FONT_MONO};
    color: {TEXT_DIM};
    font-size: 0.58rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.22rem;
}}
.kpi-value {{
    font-family: {FONT_DISPLAY};
    color: {TEXT};
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    line-height: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    text-transform: uppercase;
}}
.kpi-delta {{
    font-family: {FONT_MONO};
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 0.28rem;
    letter-spacing: 0.02em;
}}
.kpi-delta.pos {{ color: {GREEN}; }}
.kpi-delta.neg {{ color: {CRIMSON}; }}
.kpi-delta.neu {{ color: {TEXT_DIM}; }}

[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 0;
    padding: 0.65rem 0.75rem;
    min-height: 76px;
}}
[data-testid="stMetricLabel"] {{
    font-family: {FONT_MONO} !important;
    color: {TEXT_DIM} !important;
    font-size: 0.58rem !important;
    letter-spacing: 0.12em;
    text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    font-family: {FONT_DISPLAY} !important;
    color: {TEXT} !important;
    font-size: 1.45rem !important;
    font-weight: 800 !important;
}}

/* ── Scoreboard (deep dive) ───────────────────────────── */
.scoreboard {{
    display: grid;
    grid-template-columns: repeat(8, minmax(0, 1fr));
    gap: 0;
    margin: 0.2rem 0 1rem;
    background: {SURFACE};
    border: 1px solid {BORDER};
}}
.scoreboard.n3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
.scoreboard.n4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.scoreboard.n6 {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
.scoreboard.n7 {{ grid-template-columns: repeat(7, minmax(0, 1fr)); }}
.scoreboard.n8 {{ grid-template-columns: repeat(8, minmax(0, 1fr)); }}
.sb-cell {{
    padding: 0.6rem 0.7rem 0.55rem;
    border-right: 1px solid {BORDER_SUBTLE};
}}
.sb-cell:last-child {{ border-right: none; }}
.sb-k {{
    display: block;
    font-family: {FONT_MONO};
    font-size: 0.56rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin-bottom: 0.18rem;
}}
.sb-v {{
    display: block;
    font-family: {FONT_DISPLAY};
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    color: {TEXT};
    line-height: 1;
    text-transform: uppercase;
}}

.dossier {{
    margin: 0.15rem 0 0.75rem;
    padding: 0.15rem 0 0.7rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.dossier .kicker {{
    font-family: {FONT_MONO};
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: {CRIMSON};
}}
.dossier .name {{
    font-family: {FONT_DISPLAY};
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: {TEXT};
    line-height: 0.92;
    margin: 0.05rem 0 0.25rem;
}}
.dossier .line {{
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
}}
.dossier .wl {{
    font-family: {FONT_DISPLAY};
    font-size: 1.7rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    color: {TEXT};
}}
.dossier .wl .l {{ color: {TEXT_DIM}; }}
.phase-badge {{
    display: inline-block;
    padding: 0.12rem 0.45rem;
    font-family: {FONT_MONO};
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {CYAN};
    border: 1px solid rgba(110, 203, 255, 0.35);
    background: rgba(110, 203, 255, 0.08);
}}

/* ── Leaderboards ────────────────────────────────────── */
.leaderboard {{
    list-style: none;
    margin: 0;
    padding: 0;
    border: 1px solid {BORDER};
    background: {SURFACE};
}}
.lb-row {{
    display: grid;
    grid-template-columns: 2rem 1fr minmax(3.5rem, 28%) auto;
    gap: 0.55rem;
    align-items: center;
    padding: 0.42rem 0.7rem;
    border-bottom: 1px solid {BORDER_SUBTLE};
}}
.lb-row:last-child {{ border-bottom: none; }}
.lb-rank {{
    font-family: {FONT_MONO};
    font-size: 0.68rem;
    font-weight: 600;
    color: {TEXT_DIM};
}}
.lb-row:nth-child(1) .lb-rank {{ color: {AMBER}; }}
.lb-name {{
    font-family: {FONT_UI};
    color: {TEXT};
    font-size: 0.84rem;
    font-weight: 600;
}}
.lb-bar {{
    height: 4px;
    background: {BORDER_SUBTLE};
    overflow: hidden;
}}
.lb-bar > i {{
    display: block;
    height: 100%;
    background: {CRIMSON};
}}
.lb-row.neg .lb-bar > i {{ background: {TEXT_DIM}; }}
.lb-stat {{
    font-family: {FONT_MONO};
    font-size: 0.78rem;
    font-weight: 600;
    color: {TEXT};
    text-align: right;
}}

/* ── Tables ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 0;
    overflow: hidden;
    background: {SURFACE};
}}
[data-testid="stDataFrame"] * {{
    font-size: 0.78rem !important;
}}
[data-testid="stDataFrame"] th,
.dvn-scroller .col-header-cell {{
    background-color: {SURFACE_2} !important;
    color: {TEXT_DIM} !important;
    font-family: {FONT_MONO} !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em;
    border-bottom: 1px solid {BORDER} !important;
}}
[data-testid="stDataFrame"] td {{
    background-color: {BG_ELEVATED};
    color: {TEXT};
    font-family: {FONT_MONO};
    font-variant-numeric: tabular-nums;
    border-bottom: 1px solid {BORDER_SUBTLE} !important;
}}

/* ── Controls ────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {{
    border-bottom: 1px solid {BORDER};
    gap: 0;
}}
[data-testid="stTabs"] [role="tab"] {{
    font-family: {FONT_DISPLAY} !important;
    color: {TEXT_DIM} !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase !important;
    padding: 0.4rem 0.85rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {TEXT} !important;
    border-bottom-color: {CRIMSON} !important;
}}
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label {{
    font-family: {FONT_MONO} !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: {TEXT_DIM} !important;
}}
[data-testid="stSelectbox"] > div > div {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
    font-family: {FONT_DISPLAY} !important;
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    min-height: 2.4rem !important;
}}
[data-testid="stTextInput"] > div > div > input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 0 !important;
    color: {TEXT} !important;
    font-family: {FONT_MONO} !important;
}}
.stButton > button {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 0;
    font-family: {FONT_DISPLAY};
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    min-height: 2.4rem;
}}
.stButton > button:hover {{ border-color: {CRIMSON}; color: {TEXT}; }}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
    background: {CRIMSON} !important;
}}
[data-testid="stAlert"] {{
    border-radius: 0 !important;
    border: 1px solid {BORDER} !important;
    background: {SURFACE} !important;
    font-size: 0.8rem !important;
}}
hr {{ border-color: {BORDER_SUBTLE} !important; margin: 0.7rem 0 !important; }}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
    background: {SURFACE_2} !important;
    color: {TEXT} !important;
    border-radius: 0 !important;
    font-family: {FONT_MONO} !important;
}}

.empty-card {{
    background: {SURFACE};
    border: 1px dashed {BORDER};
    padding: 1.4rem 1.5rem;
    margin: 0.7rem 0 1.1rem;
}}
.empty-card h3 {{
    font-family: {FONT_DISPLAY} !important;
    font-size: 1.35rem !important;
    letter-spacing: 0.04em !important;
    color: {TEXT} !important;
    margin: 0 0 0.35rem;
}}
.empty-card p {{ color: {TEXT_MUTED}; font-size: 0.88rem; margin: 0 0 0.7rem; }}
.empty-card pre {{
    background: {BG};
    color: {TEXT};
    border: 1px solid {BORDER_SUBTLE};
    padding: 0.7rem 0.85rem;
    font-size: 0.74rem;
    font-family: {FONT_MONO};
    margin: 0;
}}

.js-plotly-plot {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER};
}}
.js-plotly-plot .plotly .modebar {{ display: none !important; }}
[data-testid="stPlotlyChart"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    padding: 0.15rem 0.2rem 0;
}}

.filter-rail {{
    display: flex;
    align-items: flex-end;
    gap: 0.75rem;
    margin: 0 0 0.85rem;
    padding: 0.55rem 0.75rem 0.65rem;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {CRIMSON};
}}

.caption-bar {{
    font-family: {FONT_MONO};
    font-size: 0.66rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {TEXT_DIM};
    margin: 0.15rem 0 0.55rem;
}}

@media (max-width: 1100px) {{
    .kpi-grid, .scoreboard, .scoreboard.n8, .scoreboard.n7, .scoreboard.n6 {{
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }}
    .masthead h1, .dossier .name {{ font-size: 1.85rem; }}
    .frame-meta {{ display: none; }}
}}
"""
