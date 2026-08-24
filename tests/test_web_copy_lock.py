"""Keep Next.js BenchOrStart copy + stubs locked to the Python shell."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fantasy.cards import load_stub_cards
from fantasy.copy import (
    COPIED,
    COPY_TEXT,
    CTA,
    DOWNLOAD_IMAGE,
    EARLY_MODEL_BADGE,
    EMAIL_ERROR,
    EMPTY_BODY,
    EMPTY_TAB,
    EMPTY_TITLE,
    FOOTER,
    HEADLINE,
    INVITE_CHIP,
    MICROCOPY,
    PRIOR_SEASON_BANNER,
    PRODUCT_NAME,
    PROMPT_LINE,
    STUB_CAPTION,
    SUBHEAD,
    SUCCESS,
    TAB_ALL,
    TAB_LABELS,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
COPY_TS = ROOT / "packages" / "card-schema" / "src" / "copy.ts"
STUB_JSON = ROOT / "packages" / "api-client" / "src" / "stub-cards.json"
WEB_APP = ROOT / "apps" / "web" / "app" / "page.tsx"
FO_APP = ROOT / "dashboard" / "app.py"
FALLBACK = ROOT / "dashboard" / "fantasy_app.py"


def test_card_schema_copy_matches_python_lock() -> None:
    ts = COPY_TS.read_text(encoding="utf-8")
    for value in (
        PRODUCT_NAME,
        PROMPT_LINE,
        HEADLINE,
        SUBHEAD,
        CTA,
        MICROCOPY,
        SUCCESS,
        FOOTER,
        EARLY_MODEL_BADGE,
        EMAIL_ERROR,
        STUB_CAPTION,
        INVITE_CHIP,
        COPY_TEXT,
        DOWNLOAD_IMAGE,
        COPIED,
        EMPTY_TAB,
        TAB_ALL,
        EMPTY_TITLE,
        EMPTY_BODY,
        PRIOR_SEASON_BANNER,
        *TAB_LABELS,
    ):
        assert value in ts
    assert PRIOR_SEASON_BANNER == "These picks are not the current season yet."
    assert "Contract Watch" not in ts


def test_web_stub_cards_match_fantasy_jsonl() -> None:
    web = json.loads(STUB_JSON.read_text(encoding="utf-8"))
    assert web == load_stub_cards()
    types = {card["recommendation_type"] for card in web}
    assert types == {"pickup", "stream", "start", "sit"}
    sit = next(card for card in web if card["recommendation_type"] == "sit")
    assert sit["edge"]["war_source"] == "approx"
    assert sit["edge"]["is_approx"] is True


def test_next_web_exists_and_fo_stays_untouched() -> None:
    assert WEB_APP.is_file()
    assert FALLBACK.is_file()
    fo = FO_APP.read_text(encoding="utf-8")
    assert "BenchOrStart" not in fo
    assert "apps/web" not in fo
    web_root = ROOT / "apps" / "web"
    web_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in web_root.rglob("*")
        if path.suffix in {".ts", ".tsx"}
        and "node_modules" not in path.parts
        and ".next" not in path.parts
    )
    assert "Contract Watch" not in web_text
    assert "filter_contract_watch_rows" not in web_text
    assert "PRIOR_SEASON_BANNER" in (web_root / "components" / "Home.tsx").read_text(
        encoding="utf-8"
    )


def test_architect_stub_contract_is_locked() -> None:
    """#140 / architect stub contract — cards 1.0, waitlist shape, sit→BENCH."""
    client = (ROOT / "packages" / "api-client" / "src" / "client.ts").read_text(
        encoding="utf-8"
    )
    present = (ROOT / "packages" / "card-schema" / "src" / "present.ts").read_text(
        encoding="utf-8"
    )
    types = (ROOT / "packages" / "card-schema" / "src" / "types.ts").read_text(
        encoding="utf-8"
    )
    home = (ROOT / "apps" / "web" / "components" / "Home.tsx").read_text(encoding="utf-8")
    waitlist = (ROOT / "apps" / "web" / "app" / "api" / "waitlist" / "route.ts").read_text(
        encoding="utf-8"
    )
    form = (ROOT / "apps" / "web" / "components" / "WaitlistForm.tsx").read_text(
        encoding="utf-8"
    )
    assert "schema_version" in json.dumps(load_stub_cards())
    assert {card["recommendation_type"] for card in load_stub_cards()} == {
        "start",
        "sit",
        "pickup",
        "stream",
    }
    assert 'sit: "BENCH"' in types
    assert "normalizeStatLine" in present
    assert "INVITE_CHIP" in home
    assert 'source: "benchorstart"' in form
    assert "created_at" in form
    assert 'WAITLIST_SOURCE = "benchorstart"' in waitlist
    assert "created_at" in waitlist
    assert "defaultSeasonYears" in client
    assert "current_season_missing" in client
    assert "PRIOR_SEASON_BANNER" in home


def test_docs_name_streamlit_as_fallback() -> None:
    guide = (ROOT / "docs" / "fantasy.md").read_text(encoding="utf-8")
    assert "apps/web" in guide
    assert "Streamlit fallback" in guide or "local fallback" in guide
    assert "NEXT_PUBLIC_API_URL" in guide
    assert "vs repl" in guide
    assert "early model" in guide


def test_product_lock_copy_and_prior_only_ship() -> None:
    """Cole owns fantasy/copy.py. Live 2026 is #131, not an apps/web ship gate."""
    ts = COPY_TS.read_text(encoding="utf-8")
    assert "unless Cole edits" in ts
    assert "Do not invent" in ts
    guide = (ROOT / "docs" / "fantasy.md").read_text(encoding="utf-8")
    assert "unless Cole edits" in guide
    assert "not a ship gate" in guide
    assert "#131" in guide
    web_readme = (ROOT / "apps" / "web" / "README.md").read_text(encoding="utf-8")
    assert "unless Cole edits" in web_readme
    assert "#131" in web_readme
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "unless Cole edits" in agents
    assert "#131" in agents
