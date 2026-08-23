"""Visual system for the Streamlit dashboard — tokens, CSS, Plotly theme."""
from __future__ import annotations

# ── Palette ──────────────────────────────────────────────────────────────────
BG = "#07090d"
BG_ELEVATED = "#0b0f16"
SURFACE = "#111822"
SURFACE_2 = "#17202c"
BORDER = "#243044"
BORDER_SUBTLE = "#1a2433"
TEXT = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
TEXT_DIM = "#64748b"
CRIMSON = "#e11d2e"
CRIMSON_DEEP = "#be123c"
CRIMSON_SOFT = "rgba(225, 29, 46, 0.14)"
CYAN = "#38bdf8"
GREEN = "#22c55e"
AMBER = "#f59e0b"
PURPLE = "#a78bfa"
ORANGE = "#fb7185"

FONT_UI = '"Plus Jakarta Sans", "IBM Plex Sans", -apple-system, BlinkMacSystemFont, sans-serif'
FONT_MONO = '"IBM Plex Mono", "JetBrains Mono", ui-monospace, monospace'

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family='IBM Plex Sans, Plus Jakarta Sans, sans-serif', color=TEXT_MUTED, size=12),
    title_font=dict(size=13, color=TEXT, family='Plus Jakarta Sans, sans-serif', weight=600),
    title=dict(x=0.0, xanchor="left", pad=dict(t=4, b=8)),
    xaxis=dict(
        gridcolor=BORDER_SUBTLE,
        linecolor=BORDER,
        tickcolor=BORDER,
        zerolinecolor=BORDER_SUBTLE,
        tickfont=dict(color=TEXT_DIM, size=11, family="IBM Plex Mono, monospace"),
        title_font=dict(color=TEXT_MUTED, size=11),
        showspikes=False,
    ),
    yaxis=dict(
        gridcolor=BORDER_SUBTLE,
        linecolor=BORDER,
        tickcolor=BORDER,
        zerolinecolor=BORDER_SUBTLE,
        tickfont=dict(color=TEXT_DIM, size=11, family="IBM Plex Mono, monospace"),
        title_font=dict(color=TEXT_MUTED, size=11),
        showspikes=False,
    ),
    legend=dict(
        bgcolor=SURFACE,
        bordercolor=BORDER,
        borderwidth=1,
        font=dict(size=11, color=TEXT),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    hoverlabel=dict(
        bgcolor=SURFACE_2,
        bordercolor=BORDER,
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=TEXT),
        align="left",
    ),
    margin=dict(t=56, b=40, l=48, r=20),
    colorway=[CRIMSON, CYAN, GREEN, AMBER, PURPLE, ORANGE, "#60a5fa"],
)

# Keep a named alias used by regression tests that exec `_PLOTLY_LAYOUT` from app.py
# and assert a dark plot background after `_chart` applies the layout.
_LEGACY_PLOTLY_LAYOUT = {
    **PLOTLY_LAYOUT,
    "paper_bgcolor": BG,
    "plot_bgcolor": BG,
}

SCATTER_MARKER = dict(size=9, opacity=0.88, line=dict(width=0.6, color=BG))

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, p, label, span {
    font-family: """ + FONT_UI + """ !important;
}
.stApp { background: """ + BG + """; color: """ + TEXT + """; }
.stApp header[data-testid="stHeader"] {
    background: transparent;
    border-bottom: none;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { display: none; }

.block-container {
    padding-top: 1.15rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1480px;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: """ + BG_ELEVATED + """;
    border-right: 1px solid """ + BORDER_SUBTLE + """;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.1rem 0.85rem 1.4rem; }
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { color: """ + TEXT_DIM + """; }

.sidebar-brand {
    padding: 0.15rem 0.35rem 1.05rem;
    border-bottom: 1px solid """ + BORDER_SUBTLE + """;
    margin-bottom: 0.85rem;
}
.sidebar-brand .mark {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}
.sidebar-brand .glyph {
    width: 34px; height: 34px;
    border-radius: 9px;
    background: linear-gradient(160deg, """ + CRIMSON + """ 0%, """ + CRIMSON_DEEP + """ 100%);
    color: #fff;
    font-size: 1.05rem;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 0 0 1px rgba(225,29,46,0.35), 0 8px 18px rgba(225,29,46,0.18);
}
.sidebar-brand h1 {
    font-family: """ + FONT_UI + """;
    font-size: 0.98rem;
    font-weight: 800;
    color: """ + TEXT + """;
    letter-spacing: -0.035em;
    margin: 0;
    line-height: 1.15;
    border: none;
    padding: 0;
    text-transform: none;
}
.sidebar-brand h1 em {
    font-style: normal;
    color: """ + CRIMSON + """;
}
.sidebar-brand small {
    display: block;
    color: """ + TEXT_DIM + """;
    font-size: 0.62rem;
    margin-top: 3px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-weight: 600;
}
.nav-group {
    color: """ + TEXT_DIM + """;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0.85rem 0.4rem 0.35rem;
}

[data-testid="stSidebar"] .stButton { margin-bottom: 2px; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent !important;
    color: """ + TEXT_MUTED + """ !important;
    border: 1px solid transparent !important;
    border-left: 2px solid transparent !important;
    border-radius: 8px !important;
    justify-content: flex-start !important;
    text-align: left !important;
    font-size: 0.84rem !important;
    font-weight: 550 !important;
    letter-spacing: -0.01em;
    padding: 0.42rem 0.7rem !important;
    min-height: 2.15rem !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.035) !important;
    color: """ + TEXT + """ !important;
    border-color: """ + BORDER_SUBTLE + """ !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    background: """ + CRIMSON_SOFT + """ !important;
    color: """ + TEXT + """ !important;
    border-color: rgba(225, 29, 46, 0.22) !important;
    border-left: 2px solid """ + CRIMSON + """ !important;
    font-weight: 700 !important;
}

.sidebar-status {
    margin-top: 1.15rem;
    padding: 0.85rem 0.75rem 0.15rem;
    border-top: 1px solid """ + BORDER_SUBTLE + """;
    color: """ + TEXT_DIM + """;
    font-size: 0.72rem;
    line-height: 1.55;
}
.sidebar-status .status-row { display: flex; justify-content: space-between; gap: 0.5rem; margin-bottom: 0.35rem; }
.sidebar-status strong { color: """ + TEXT + """; font-weight: 600; }
.status-pill {
    display: inline-flex; align-items: center; gap: 0.35rem;
    margin-top: 0.55rem;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.status-pill.live { background: rgba(34,197,94,0.12); color: """ + GREEN + """; }
.status-pill.setup { background: rgba(245,158,11,0.12); color: """ + AMBER + """; }
.status-pill .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.15);
}

/* ── Type ────────────────────────────────────────────── */
.page-kicker {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: """ + CRIMSON + """;
    margin-bottom: 0.2rem;
}
h1 {
    font-family: """ + FONT_UI + """ !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: """ + TEXT + """ !important;
    letter-spacing: -0.04em !important;
    border-bottom: none !important;
    padding-bottom: 0 !important;
    margin-bottom: 0.2rem !important;
    line-height: 1.15 !important;
}
h2 {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    color: """ + TEXT + """ !important;
    letter-spacing: -0.015em !important;
    margin-top: 0.35rem !important;
}
h3 {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    color: """ + TEXT_MUTED + """ !important;
    text-transform: uppercase;
    letter-spacing: 0.1em !important;
}
.page-lead, .stCaption, [data-testid="stCaptionContainer"] {
    color: """ + TEXT_MUTED + """ !important;
    font-size: 0.88rem !important;
    line-height: 1.5;
}
.war-note {
    color: """ + TEXT_DIM + """;
    font-size: 0.78rem;
    margin: 0.15rem 0 0.85rem;
}

.panel-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.75rem;
    margin: 0.15rem 0 0.55rem;
}
.panel-head .title {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: """ + TEXT_MUTED + """;
}
.panel-head .hint { font-size: 0.75rem; color: """ + TEXT_DIM + """; }

/* ── KPI strip ───────────────────────────────────────── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0.35rem 0 1.15rem;
}
.kpi-card {
    background: linear-gradient(180deg, """ + SURFACE + """ 0%, """ + BG_ELEVATED + """ 100%);
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    padding: 0.75rem 0.85rem 0.7rem;
    min-height: 86px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
.kpi-label {
    color: """ + TEXT_DIM + """;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.28rem;
}
.kpi-value {
    color: """ + TEXT + """;
    font-family: """ + FONT_MONO + """;
    font-size: 1.18rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.kpi-delta { font-size: 0.74rem; font-weight: 650; margin-top: 0.28rem; font-family: """ + FONT_MONO + """; }
.kpi-delta.pos { color: """ + GREEN + """; }
.kpi-delta.neg { color: """ + CRIMSON + """; }
.kpi-delta.neu { color: """ + TEXT_DIM + """; }

[data-testid="stMetric"] {
    background: linear-gradient(180deg, """ + SURFACE + """ 0%, """ + BG_ELEVATED + """ 100%);
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    min-height: 88px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
[data-testid="stMetricLabel"] {
    color: """ + TEXT_DIM + """ !important;
    font-size: 0.64rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    color: """ + TEXT + """ !important;
    font-family: """ + FONT_MONO + """ !important;
    font-size: 1.28rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
}
[data-testid="stMetricDelta"] { font-size: 0.76rem !important; font-weight: 650; }

/* ── Leaderboards ────────────────────────────────────── */
.leaderboard {
    list-style: none;
    margin: 0;
    padding: 0.15rem 0 0;
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    overflow: hidden;
    background: """ + SURFACE + """;
}
.lb-row {
    display: grid;
    grid-template-columns: 28px 1fr auto;
    gap: 0.65rem;
    align-items: center;
    padding: 0.55rem 0.85rem;
    border-bottom: 1px solid """ + BORDER_SUBTLE + """;
}
.lb-row:last-child { border-bottom: none; }
.lb-row:nth-child(odd) { background: rgba(255,255,255,0.015); }
.lb-rank {
    font-family: """ + FONT_MONO + """;
    font-size: 0.72rem;
    font-weight: 600;
    color: """ + TEXT_DIM + """;
}
.lb-row:nth-child(1) .lb-rank { color: """ + AMBER + """; }
.lb-name { color: """ + TEXT + """; font-size: 0.86rem; font-weight: 600; letter-spacing: -0.01em; }
.lb-stat {
    font-family: """ + FONT_MONO + """;
    font-size: 0.8rem;
    font-weight: 600;
    color: """ + TEXT_MUTED + """;
}

/* ── Tables ──────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid """ + BORDER + """;
    border-radius: 10px;
    overflow: hidden;
    background: """ + SURFACE + """;
}
[data-testid="stDataFrame"] th,
.dvn-scroller .col-header-cell {
    background-color: """ + SURFACE_2 + """ !important;
    color: """ + TEXT_DIM + """ !important;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid """ + BORDER + """ !important;
}
[data-testid="stDataFrame"] td {
    background-color: """ + BG_ELEVATED + """;
    color: """ + TEXT + """;
    font-size: 0.82rem;
    border-bottom: 1px solid """ + BORDER_SUBTLE + """ !important;
    font-variant-numeric: tabular-nums;
    font-family: """ + FONT_MONO + """;
}

/* ── Controls ────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid """ + BORDER + """; gap: 0.15rem; }
[data-testid="stTabs"] [role="tab"] {
    color: """ + TEXT_MUTED + """ !important;
    font-size: 0.8rem !important;
    font-weight: 650 !important;
    padding: 0.5rem 0.95rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: """ + TEXT + """ !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: """ + TEXT + """ !important;
    border-bottom-color: """ + CRIMSON + """ !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] input {
    background-color: """ + SURFACE + """ !important;
    border-color: """ + BORDER + """ !important;
    color: """ + TEXT + """ !important;
    font-size: 0.86rem !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: """ + SURFACE_2 + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER + """;
    font-size: 0.82rem;
    font-weight: 650;
    border-radius: 8px;
    min-height: 2.35rem;
}
.stButton > button:hover { background: """ + BORDER + """; border-color: """ + TEXT_DIM + """; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: """ + CRIMSON + """ !important;
}
[data-testid="stExpander"] {
    border: 1px solid """ + BORDER + """ !important;
    border-radius: 10px !important;
    background: """ + SURFACE + """ !important;
}
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.84rem !important;
    border: 1px solid """ + BORDER + """ !important;
    background: """ + SURFACE + """ !important;
}
hr { border-color: """ + BORDER_SUBTLE + """ !important; margin: 1rem 0 !important; }
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: """ + SURFACE_2 + """ !important;
    color: """ + TEXT + """ !important;
    border-radius: 5px !important;
}

.toolbar {
    background: """ + SURFACE + """;
    border: 1px solid """ + BORDER + """;
    border-radius: 12px;
    padding: 0.15rem 0.35rem 0.05rem;
    margin: 0.35rem 0 1rem;
}

.dossier-title {
    display: flex; align-items: baseline; gap: 0.75rem;
    margin: 0.4rem 0 0.85rem;
}
.dossier-title h2 {
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    text-transform: none !important;
    color: """ + TEXT + """ !important;
    margin: 0 !important;
}
.phase-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: rgba(56,189,248,0.12);
    color: """ + CYAN + """;
    border: 1px solid rgba(56,189,248,0.25);
}

.empty-card {
    background: """ + SURFACE + """;
    border: 1px dashed """ + BORDER + """;
    border-radius: 12px;
    padding: 1.45rem 1.55rem;
    margin: 0.85rem 0 1.25rem;
}
.empty-card h3 {
    color: """ + TEXT + """ !important;
    text-transform: none !important;
    letter-spacing: -0.02em !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    margin: 0 0 0.4rem;
}
.empty-card p { color: """ + TEXT_MUTED + """; font-size: 0.9rem; margin: 0 0 0.8rem; line-height: 1.55; }
.empty-card pre {
    background: """ + BG + """;
    color: """ + TEXT + """;
    border: 1px solid """ + BORDER_SUBTLE + """;
    border-radius: 8px;
    padding: 0.8rem 0.95rem;
    font-size: 0.76rem;
    overflow-x: auto;
    margin: 0;
    font-family: """ + FONT_MONO + """;
}

@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 720px) {
    .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""
