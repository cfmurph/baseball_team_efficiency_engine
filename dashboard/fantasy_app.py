"""BenchOrStart — waitlist + share-card shell (Phase 0 / #112).

Separate entrypoint from the front-office GM dashboard::

    streamlit run dashboard/fantasy_app.py

Reads published cards via ``resolve_artifact`` / ``ARTIFACTS_URI``:
``current/fantasy/cards.jsonl``, then optional ``fantasy/cards.jsonl``.
Player CSVs use the same #105 loaders as the FO dashboard. Missing
feeds show an empty state; bundled samples stay labeled as samples.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from src.baseball_analytics.config import load_artifact_settings
from src.baseball_analytics.storage import resolve_artifact

from fantasy.cards import (
    SOURCE_MISSING,
    load_share_cards,
    load_stub_cards,
    present_cards,
    resolve_player_artifacts,
    share_card_html,
)
from fantasy.copy import (
    CTA,
    EMPTY_BODY,
    EMPTY_TITLE,
    FOOTER,
    HEADLINE,
    MICROCOPY,
    PRODUCT_NAME,
    STUB_CAPTION,
    SUBHEAD,
    SUCCESS,
)
from fantasy.waitlist import capture_signup

st.set_page_config(
    page_title=PRODUCT_NAME,
    page_icon="⚾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

_ARTIFACT_SETTINGS = load_artifact_settings(str(_ROOT / "config/settings.yaml"))


@st.cache_data(ttl=300)
def _load_cards() -> tuple[list[dict], str]:
    # Same published player CSVs as FO; unused for ranking in this shell.
    resolve_player_artifacts(_ARTIFACT_SETTINGS)
    feed = load_share_cards(_ARTIFACT_SETTINGS)
    return feed.cards, feed.source


st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp { background-color: #0d1117; }
.block-container {
    padding-top: 2.2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 760px;
}
.bos-hero { text-align: center; margin: 0 0 1.6rem; }
.bos-brand {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #f85149;
    margin-bottom: 0.7rem;
}
.bos-hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.035em !important;
    line-height: 1.2 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 0.65rem !important;
}
.bos-hero p {
    color: #b1bac4;
    font-size: 1.02rem;
    line-height: 1.5;
    margin: 0 auto;
    max-width: 34rem;
}
.bos-micro {
    color: #8b949e;
    font-size: 0.8rem;
    text-align: center;
    margin: 0.35rem 0 0;
}
.bos-success {
    background: #122117;
    border: 1px solid #238636;
    color: #3fb950;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    text-align: center;
    font-weight: 600;
    margin: 0.6rem 0 0.2rem;
}
.bos-error {
    background: #2a1215;
    border: 1px solid #f85149;
    color: #f85149;
    border-radius: 8px;
    padding: 0.65rem 1rem;
    text-align: center;
    margin: 0.6rem 0 0.2rem;
}
.bos-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: 4px solid var(--bos-tone, #f85149);
    border-radius: 14px;
    padding: 1.35rem 1.4rem 1.15rem;
    margin: 0.75rem 0 1rem;
}
.bos-card-featured { padding: 1.6rem 1.5rem 1.25rem; }
.bos-wordmark {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f85149;
}
.bos-prompt {
    font-size: 1.35rem;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.03em;
    margin: 0.45rem 0 0.55rem;
}
.bos-label {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #0d1117;
    background: var(--bos-tone, #58a6ff);
    border-radius: 999px;
    padding: 0.22rem 0.65rem;
}
.bos-badge {
    display: inline-block;
    margin-left: 0.4rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #d29922;
    border: 1px solid #d29922;
    border-radius: 999px;
    padding: 0.16rem 0.5rem;
}
.bos-rank {
    color: #8b949e;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.55rem 0 0.15rem;
}
.bos-headline {
    font-size: 1.55rem !important;
    font-weight: 800 !important;
    color: #e6edf3 !important;
    letter-spacing: -0.03em !important;
    border: none !important;
    padding: 0 !important;
    margin: 0.35rem 0 0.2rem !important;
}
.bos-sub { color: #b1bac4; font-size: 0.98rem; font-weight: 500; }
.bos-stat {
    color: #e6edf3;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0.7rem 0 0.35rem;
}
.bos-reason { color: #b1bac4; font-size: 0.95rem; line-height: 1.45; margin: 0.4rem 0 0; }
.bos-asof { color: #8b949e; font-size: 0.72rem; margin-top: 0.85rem; }
.bos-caption { color: #8b949e; font-size: 0.78rem; text-align: center; margin: 0.25rem 0 1rem; }
.bos-empty {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.3rem;
    margin: 1.4rem 0 0.8rem;
    text-align: center;
}
.bos-empty h2 {
    color: #e6edf3 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 0 0.4rem !important;
    text-transform: none !important;
    letter-spacing: -0.02em !important;
}
.bos-empty p { color: #b1bac4; font-size: 0.92rem; margin: 0; line-height: 1.45; }
.bos-foot {
    color: #8b949e;
    font-size: 0.78rem;
    text-align: center;
    margin-top: 2rem;
}
[data-testid="stTextInput"] > div > div > input {
    background-color: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}
.stButton > button {
    background: #bf1c20;
    color: #fff;
    border: 1px solid #bf1c20;
    font-weight: 700;
    border-radius: 8px;
    min-height: 2.6rem;
    width: 100%;
}
.stButton > button:hover { background: #f85149; border-color: #f85149; }
.stForm { border: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="bos-hero">
      <div class="bos-brand">{html.escape(PRODUCT_NAME)}</div>
      <h1>{html.escape(HEADLINE)}</h1>
      <p>{html.escape(SUBHEAD)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("waitlist", clear_on_submit=False):
    email = st.text_input("Email", placeholder="you@email.com", label_visibility="collapsed")
    submitted = st.form_submit_button(CTA)
    if submitted:
        result = capture_signup(email)
        if result.ok:
            st.markdown(f'<div class="bos-success">{html.escape(SUCCESS)}</div>', unsafe_allow_html=True)
        else:
            message = result.error or "Enter a valid email."
            st.markdown(f'<div class="bos-error">{html.escape(message)}</div>', unsafe_allow_html=True)
st.markdown(f'<p class="bos-micro">{html.escape(MICROCOPY)}</p>', unsafe_allow_html=True)

cards, source = _load_cards()
live_cards = [] if source == SOURCE_MISSING else cards
views = present_cards(live_cards)

if not views:
    st.markdown(
        f"""
        <div class="bos-empty" role="status">
          <h2>{html.escape(EMPTY_TITLE)}</h2>
          <p>{html.escape(EMPTY_BODY)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    views = present_cards(load_stub_cards())
    if views:
        st.markdown(share_card_html(views[0], featured=True), unsafe_allow_html=True)
        for view in views[1:]:
            st.markdown(share_card_html(view), unsafe_allow_html=True)
        st.markdown(f'<p class="bos-caption">{html.escape(STUB_CAPTION)}</p>', unsafe_allow_html=True)
else:
    st.markdown(share_card_html(views[0], featured=True), unsafe_allow_html=True)
    for view in views[1:]:
        st.markdown(share_card_html(view), unsafe_allow_html=True)

st.markdown(f'<p class="bos-foot">{html.escape(FOOTER)}</p>', unsafe_allow_html=True)
