from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fantasy.cards import CARD_LAKE_KEY, RUN_CARD_TEMPLATE, SOURCE_MISSING, CardLoad
from fantasy.copy import (
    COPY_TEXT,
    CTA,
    DOWNLOAD_IMAGE,
    EMPTY_BODY,
    EMPTY_TITLE,
    FOOTER,
    HEADLINE,
    INVITE_CHIP,
    MICROCOPY,
    PRIOR_SEASON_BANNER,
    PRODUCT_NAME,
    SUBHEAD,
    SUCCESS,
)

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "fantasy_app.py"
ROOT = APP_PATH.resolve().parents[1]
GM_APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _is_local_import(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom) and node.module:
        module = node.module
        if module.startswith(("src", "dashboard", "fantasy")):
            return module
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith(("src", "dashboard", "fantasy")):
                return alias.name
    return None


def _is_path_bootstrap(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and "sys.path.insert" in ast.unparse(node)


def test_fantasy_entrypoint_bootstraps_before_local_imports() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    saw_bootstrap = False
    for node in tree.body:
        if _is_path_bootstrap(node):
            saw_bootstrap = True
            continue
        module = _is_local_import(node)
        if module is not None:
            assert saw_bootstrap, f"{module} is imported before the sys.path bootstrap"
    assert saw_bootstrap


def test_fantasy_app_uses_shared_cards_jsonl_and_marketing_copy() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "resolve_player_artifacts" in source
    assert CARD_LAKE_KEY in source
    assert RUN_CARD_TEMPLATE in source
    assert "Dated JSON filenames are ignored" in source
    for name in (
        "PRODUCT_NAME",
        "HEADLINE",
        "SUBHEAD",
        "CTA",
        "MICROCOPY",
        "SUCCESS",
        "FOOTER",
        "EMPTY_TITLE",
        "EMPTY_BODY",
        "PRIOR_SEASON_BANNER",
    ):
        assert name in source
    assert PRIOR_SEASON_BANNER == "These picks are not the current season yet."
    assert HEADLINE.startswith("Know who to start")
    assert CTA == "Get early access"
    assert PRODUCT_NAME == "BenchOrStart"


def test_gm_dashboard_not_rewritten_for_fantasy() -> None:
    source = GM_APP.read_text(encoding="utf-8")
    assert "BenchOrStart" not in source
    assert "fantasy_app" not in source
    assert "current/fantasy/cards.jsonl" not in source


def test_fantasy_app_soft_launch_layout_and_share_actions() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert source.index("views = present_cards") < source.index('st.form("waitlist"')
    assert "st.tabs" in source
    assert "share_blurb" in source
    assert "render_share_card_png" in source
    assert "import streamlit.components" not in source
    assert "unsafe_allow_javascript" in source
    assert "INVITE_CHIP" in source
    assert "TAB_LABELS" in source
    assert "EMPTY_TITLE" in source
    assert COPY_TEXT == "Copy text"
    assert DOWNLOAD_IMAGE == "Download image"
    assert INVITE_CHIP == "Invite only"
    chrome = source.lower()
    assert "efficiency engine" not in chrome
    assert "front office" not in chrome


def test_fantasy_app_imports_without_pythonpath() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = r"""
import sys
from pathlib import Path

script_dir = Path("dashboard").resolve()
root = script_dir.parent
sys.path = [str(script_dir)] + [p for p in sys.path if Path(p).resolve() != root]

_ROOT = Path("dashboard/fantasy_app.py").resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fantasy.cards import CARD_LAKE_KEY, load_stub_cards
from fantasy.copy import PRODUCT_NAME
from src.baseball_analytics.storage import resolve_artifact

assert CARD_LAKE_KEY == "current/fantasy/cards.jsonl"
assert PRODUCT_NAME == "BenchOrStart"
assert callable(resolve_artifact)
assert len(load_stub_cards()) == 4
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_fantasy_app_wires_prior_season_banner_helpers() -> None:
    source = APP_PATH.read_text(encoding="utf-8")
    assert "is_prior_only_publish" in source
    assert "load_metrics_manifest" in source
    assert "max_season_from_cards" in source
    assert "seasons_from_manifest" in source
    assert "live_feed=True" in source
    assert "SOURCE_MISSING" in source


def test_fantasy_apptest_stubs_do_not_show_prior_season_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(ROOT)
    import streamlit as st

    st.cache_data.clear()
    monkeypatch.setattr(
        "fantasy.cards.load_share_cards",
        lambda *args, **kwargs: CardLoad(cards=[], source=SOURCE_MISSING),
    )
    monkeypatch.setattr(
        "dashboard.data.load_metrics_manifest",
        lambda: {"current_season_missing": True, "active_season": 2026},
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not at.exception
    info_text = " ".join(str(item.value) for item in at.info)
    assert PRIOR_SEASON_BANNER not in info_text


def test_fantasy_apptest_live_prior_only_shows_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    import streamlit as st

    st.cache_data.clear()
    live = [
        {
            "schema_version": "1.0",
            "card_id": "live-start-1",
            "recommendation_type": "start",
            "as_of_date": "2026-08-23",
            "season": 2024,
            "player": {"player_id": "judgeaa01", "name": "Aaron Judge", "position": "OF", "team": "NYY"},
            "edge": {
                "vs_replacement": 3.4,
                "war": 6.1,
                "war_source": "bbref",
                "is_approx": False,
                "confidence": 0.91,
            },
            "rank": {"among_rec_type": 1},
            "reason": "Lock him in.",
            "share": {"stat_line": "+3.4 edge · 91% conf"},
        }
    ]
    monkeypatch.setattr(
        "fantasy.cards.load_share_cards",
        lambda *args, **kwargs: CardLoad(cards=live, source="local"),
    )
    monkeypatch.setattr(
        "dashboard.data.load_metrics_manifest",
        lambda: {
            "current_season_missing": True,
            "active_season": 2026,
            "as_of_date": "2026-08-23",
        },
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    assert not at.exception
    info_text = " ".join(str(item.value) for item in at.info)
    assert PRIOR_SEASON_BANNER in info_text
