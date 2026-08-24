"""Keep Next.js BenchOrStart copy + stubs locked to the Python shell."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import fantasy.copy as copy_mod
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


_TS_ARRAY = re.compile(r"export const (\w+)\s*=\s*\[([^\]]+)\]")
_TS_STRING = re.compile(r'export const (\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', re.MULTILINE)


def _ts_copy_exports(text: str) -> dict[str, object]:
    arrays = {
        name: tuple(re.findall(r'"([^"]*)"', body))
        for name, body in _TS_ARRAY.findall(text)
    }
    strings = {
        name: value
        for name, value in _TS_STRING.findall(text)
        if name not in arrays
    }
    return {**strings, **arrays}


def test_card_schema_copy_matches_python_lock() -> None:
    ts = COPY_TS.read_text(encoding="utf-8")
    parsed = _ts_copy_exports(ts)
    # Cole / Product: these faces are VERBATIM — including empty states.
    assert parsed["PRODUCT_NAME"] == PRODUCT_NAME
    assert parsed["HEADLINE"] == HEADLINE
    assert parsed["CTA"] == CTA
    assert parsed["TAB_LABELS"] == tuple(TAB_LABELS)
    assert parsed["TAB_ALL"] == TAB_ALL
    assert parsed["FOOTER"] == FOOTER
    assert parsed["EMPTY_TITLE"] == EMPTY_TITLE
    assert parsed["EMPTY_BODY"] == EMPTY_BODY
    assert parsed["EMPTY_TAB"] == EMPTY_TAB
    for name in dir(copy_mod):
        if name.startswith("_") or not name.isupper():
            continue
        value = getattr(copy_mod, name)
        if isinstance(value, str):
            assert parsed[name] == value, name
        elif isinstance(value, tuple):
            assert parsed[name] == value, name
    assert PRIOR_SEASON_BANNER == "These picks are not the current season yet."
    assert "Contract Watch" not in ts
    assert "VERBATIM" in ts


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
    home = (web_root / "components" / "Home.tsx").read_text(encoding="utf-8")
    assert "PRIOR_SEASON_BANNER" in home
    for symbol in (
        "PRODUCT_NAME",
        "HEADLINE",
        "CTA",
        "FOOTER",
        "TAB_ALL",
        "TAB_LABELS",
        "EMPTY_TITLE",
        "EMPTY_BODY",
        "EMPTY_TAB",
    ):
        assert symbol in home
    assert EMPTY_TITLE not in home
    assert EMPTY_BODY not in home
    assert HEADLINE not in home


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
    assert "parseSeasonWindow" in client
    assert "#144" in client
    assert "PRIOR_SEASON_BANNER" in home
    assert "ArtifactSource" in types
    assert 'SeasonWindow = number[]' in types
    openapi = (ROOT / "services" / "api" / "openapi.yaml").read_text(encoding="utf-8")
    for field in (
        "as_of",
        "active_season",
        "current_season_missing",
        "season_window",
        "seasons_present",
        "schema_version",
    ):
        assert field in openapi
        assert field in types


def test_docs_name_streamlit_as_fallback() -> None:
    guide = (ROOT / "docs" / "fantasy.md").read_text(encoding="utf-8")
    assert "apps/web" in guide
    assert "Streamlit fallback" in guide or "local fallback" in guide
    assert "NEXT_PUBLIC_API_URL" in guide
    assert "vs repl" in guide
    assert "early model" in guide
    assert "#144" in guide


def test_product_lock_copy_and_prior_only_ship() -> None:
    """Cole owns fantasy/copy.py. Live 2026 is #131, not an apps/web ship gate."""
    ts = COPY_TS.read_text(encoding="utf-8")
    assert "VERBATIM" in ts
    assert "do not rewrite" in ts
    guide = (ROOT / "docs" / "fantasy.md").read_text(encoding="utf-8")
    assert "VERBATIM" in guide
    assert "not a ship gate" in guide
    assert "#131" in guide
    web_readme = (ROOT / "apps" / "web" / "README.md").read_text(encoding="utf-8")
    assert "VERBATIM" in web_readme
    assert "#131" in web_readme
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "VERBATIM" in agents or "unless Cole edits" in agents
    assert "#131" in agents
