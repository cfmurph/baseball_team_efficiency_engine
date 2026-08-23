"""BenchOrStart — waitlist + share-card shell (Phase 0 / #112).

Separate entrypoint from the front-office GM dashboard::

    streamlit run dashboard/fantasy_app.py

Reads locked ``current/fantasy/cards.jsonl`` (schema 1.0) via
``resolve_artifact`` / ``ARTIFACTS_URI``. ``fantasy_cards_*.json`` is
fallback only. Missing feeds show an empty state plus labeled stubs.
This entrypoint is not a page in the FO GM app.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from src.baseball_analytics.config import load_artifact_settings
from src.baseball_analytics.storage import resolve_artifact

from fantasy.card_image import render_share_card_png
from fantasy.cards import (
    SOURCE_MISSING,
    load_share_cards,
    load_stub_cards,
    present_cards,
    resolve_player_artifacts,
    share_card_html,
)
from fantasy.copy import (
    COPIED,
    COPY_TEXT,
    CTA,
    EMPTY_BODY,
    EMPTY_TITLE,
    FOOTER,
    HEADLINE,
    INVITE_CHIP,
    MICROCOPY,
    PRODUCT_NAME,
    STUB_CAPTION,
    SUBHEAD,
    SUCCESS,
    TAB_ALL,
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
.bos-brandrow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.55rem;
    margin-bottom: 0.7rem;
}
.bos-brand {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #f85149;
}
.bos-chip {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #d29922;
    border: 1px solid #3d3420;
    background: #221c10;
    border-radius: 999px;
    padding: 0.16rem 0.55rem;
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
.stDownloadButton > button {
    background: #161b22;
    color: #e6edf3;
    border: 1px solid #30363d;
    font-weight: 700;
    border-radius: 8px;
    min-height: 2.4rem;
    width: 100%;
}
.stDownloadButton > button:hover {
    background: #21262d;
    border-color: #8b949e;
    color: #fff;
}
.stForm { border: none !important; }
.bos-actions { margin: -0.35rem 0 1.15rem; }
</style>
""",
    unsafe_allow_html=True,
)

def _copy_text_button(text: str, *, key: str) -> None:
    """Clipboard control via ``st.html`` so the click runs in page context."""
    payload = json.dumps(text)
    label = json.dumps(COPY_TEXT)
    done = json.dumps(COPIED)
    st.html(
        f"""
        <button id="{html.escape(key)}" type="button" style="
            width:100%;
            min-height:2.4rem;
            background:#161b22;
            color:#e6edf3;
            border:1px solid #30363d;
            border-radius:8px;
            font-weight:700;
            font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            font-size:0.95rem;
            cursor:pointer;
        ">{html.escape(COPY_TEXT)}</button>
        <script>
        (function() {{
          const btn = document.getElementById({json.dumps(key)});
          if (!btn) {{ return; }}
          const text = {payload};
          btn.addEventListener("click", async () => {{
            let ok = false;
            try {{
              if (navigator.clipboard && navigator.clipboard.writeText) {{
                await navigator.clipboard.writeText(text);
                ok = true;
              }}
            }} catch (err) {{ ok = false; }}
            if (!ok) {{
              const ta = document.createElement("textarea");
              ta.value = text;
              ta.setAttribute("readonly", "");
              ta.style.position = "fixed";
              ta.style.left = "-9999px";
              document.body.appendChild(ta);
              ta.select();
              try {{ ok = document.execCommand("copy"); }} catch (err) {{ ok = false; }}
              document.body.removeChild(ta);
            }}
            if (ok) {{
              btn.innerText = {done};
              setTimeout(() => {{ btn.innerText = {label}; }}, 1600);
            }}
          }});
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def _widget_key(view: ShareCardView, suffix: str, index: int, tab: str) -> str:
    token = view.card_id or f"{view.recommendation_type}-{index}-{view.headline}"
    raw = f"bos-{tab}-{token}-{suffix}"
    return re.sub(r"[^A-Za-z0-9_-]+", "-", raw)


def _render_card(view: ShareCardView, *, featured: bool, index: int, tab: str) -> None:
    st.markdown(share_card_html(view, featured=featured), unsafe_allow_html=True)
    blurb = share_blurb(view)
    left, right = st.columns(2)
    with left:
        _copy_text_button(blurb, key=_widget_key(view, "copy", index, tab))
    with right:
        st.download_button(
            DOWNLOAD_IMAGE,
            data=render_share_card_png(view),
            file_name=card_share_filename(view),
            mime="image/png",
            key=_widget_key(view, "png", index, tab),
        )


def _render_card_list(views: list[ShareCardView], *, tab: str) -> None:
    if not views:
        st.markdown(f'<p class="bos-caption">{html.escape(EMPTY_TAB)}</p>', unsafe_allow_html=True)
        return
    for index, view in enumerate(views):
        _render_card(view, featured=index == 0, index=index, tab=tab)


st.markdown(
    f"""
    <div class="bos-hero">
      <div class="bos-brandrow">
        <div class="bos-brand">{html.escape(PRODUCT_NAME)}</div>
        <span class="bos-chip">{html.escape(INVITE_CHIP)}</span>
      </div>
      <h1>{html.escape(HEADLINE)}</h1>
      <p>{html.escape(SUBHEAD)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cards, source = _load_cards()
views = present_cards(cards)

if views:
    tab_labels = [TAB_ALL, *TAB_LABELS]
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        _render_card_list(views, tab=TAB_ALL)
    for tab, label in zip(tabs[1:], TAB_LABELS):
        with tab:
            _render_card_list(cards_for_label(views, label), tab=label)
    if source == "stub":
        st.markdown(f'<p class="bos-caption">{html.escape(STUB_CAPTION)}</p>', unsafe_allow_html=True)

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
