"""Streamlit AppTest: every sidebar page boots without exception."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ast

import pandas as pd

from dashboard.helpers import NAV_PAGES, PRIOR_SEASON_TABLE_NOTE, nav_labels
from dashboard.state import SEASON_YEAR
from dashboard import ui as ui_mod

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dashboard" / "app.py"


def _fail_if_exception(at: AppTest, label: str) -> None:
    if len(at.exception) == 0:
        return
    details = []
    for exc in at.exception:
        msg = getattr(exc, "message", None) or getattr(exc, "value", str(exc))
        stack = getattr(exc, "stack_trace", None) or []
        details.append(f"{msg}\n{''.join(stack)}")
    pytest.fail(f"{label} raised:\n" + "\n---\n".join(details))


def _nav_button(at: AppTest, label: str):
    """Numbered rail buttons are labeled ``01  Overview``, keyed ``nav_<key>``."""
    page = next((p for p in NAV_PAGES if p["label"] == label), None)
    assert page is not None, f"unknown nav label {label}"
    key = f"nav_{page['key']}"
    for button in at.sidebar.button:
        if getattr(button, "key", None) == key or label in str(button.label):
            return button
    pytest.fail(f"No sidebar nav button for {label} (key={key})")


def test_all_sidebar_pages_boot_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Would have caught #103: missing _status / page_* aliases / NameError on boot."""
    monkeypatch.chdir(ROOT)
    labels = nav_labels()
    assert labels, "nav_labels() must list every sidebar page"

    at = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    _fail_if_exception(at, "initial boot")

    seen = [str(button.label) for button in at.sidebar.button]
    for label in labels:
        assert any(label in text for text in seen), f"{label} missing from rail {seen}"

    for label in labels:
        _nav_button(at, label).click().run()
        _fail_if_exception(at, label)


_REQUIRED_UI_EXPORTS = (
    "inject_theme",
    "page_header",
    "empty_state",
    "season_picker",
    "salary_note",
    "render_sidebar",
    "render_app_frame",
    "SCATTER_MARKER",
)


def test_contract_watch_apptest_keeps_2026_nan_salary(monkeypatch: pytest.MonkeyPatch) -> None:
    """AppTest: SDIO overlay row with no Lahman salary still renders on Contract Watch."""
    monkeypatch.chdir(ROOT)
    players = pd.DataFrame(
        [
            {
                "name_full": "Juan Soto",
                "year_id": 2026,
                "team_name": "Mets",
                "player_type": "batter",
                "player_war": 4.1,
                "war_source": "approx",
                "salary": float("nan"),
                "surplus_value": float("nan"),
                "contract_label": None,
                "pa": 400,
                "ip": 0.0,
            }
        ]
    )
    monkeypatch.setattr(
        "dashboard.views.contracts.load_player_season_metrics",
        lambda: players,
    )
    monkeypatch.setattr(
        "dashboard.views.roster.load_player_season_metrics",
        lambda: players,
    )
    monkeypatch.setattr(
        "dashboard.views.contracts.load_metrics_manifest",
        lambda: {
            "current_season_missing": True,
            "active_season": 2026,
            "as_of_date": "2026-08-23",
        },
    )
    monkeypatch.setattr(
        "dashboard.views.roster.load_metrics_manifest",
        lambda: {
            "current_season_missing": True,
            "active_season": 2026,
            "as_of_date": "2026-08-23",
        },
    )

    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.session_state[SEASON_YEAR] = 2026
    at.session_state["contracts_season_filter"] = 2026
    at.run()
    _fail_if_exception(at, "initial boot")
    _nav_button(at, "Contract Watch").click().run()
    _fail_if_exception(at, "Contract Watch")

    frames = [element.value for element in at.dataframe]
    assert frames, "Contract Watch should render a table"
    names = set()
    for frame in frames:
        if frame is not None and "name_full" in getattr(frame, "columns", []):
            names.update(frame["name_full"].astype(str).tolist())
    assert "Juan Soto" in names
    info_text = " ".join(str(item.value) for item in at.info)
    assert PRIOR_SEASON_TABLE_NOTE in info_text


def test_roster_lab_apptest_shows_prior_season_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    players = pd.DataFrame(
        [
            {
                "name_full": "Aaron Judge",
                "year_id": 2024,
                "team_name": "Yankees",
                "player_type": "batter",
                "player_war": 10.8,
                "war_source": "real",
                "salary": float("nan"),
                "surplus_value": float("nan"),
                "pa": 700,
                "ip": 0.0,
            }
        ]
    )
    monkeypatch.setattr("dashboard.views.roster.load_player_season_metrics", lambda: players)
    monkeypatch.setattr("dashboard.views.roster.load_sr_player_metrics", lambda: None)
    monkeypatch.setattr(
        "dashboard.views.roster.load_metrics_manifest",
        lambda: {
            "current_season_missing": True,
            "active_season": 2026,
            "as_of_date": "2026-08-23",
        },
    )
    at = AppTest.from_file(str(APP_PATH), default_timeout=15)
    at.session_state[SEASON_YEAR] = 2024
    at.run()
    _fail_if_exception(at, "initial boot")
    _nav_button(at, "Roster Lab").click().run()
    _fail_if_exception(at, "Roster Lab")
    info_text = " ".join(str(item.value) for item in at.info)
    assert PRIOR_SEASON_TABLE_NOTE in info_text


def test_ui_module_exports_chrome_app_uses() -> None:
    """Incomplete views split: app.py imported `ui` that did not exist / export chrome."""
    missing = [name for name in _REQUIRED_UI_EXPORTS if not hasattr(ui_mod, name)]
    assert not missing, f"dashboard.ui missing {missing}"


def test_app_binds_ui_before_any_use() -> None:
    """Regression for NameError: ui is not defined after a partial merge."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    ui_bound = False
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname == "ui" or alias.name == "dashboard.ui":
                    ui_bound = True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "dashboard" and any(a.name == "ui" for a in node.names):
                ui_bound = True
            if node.module == "dashboard.ui":
                ui_bound = True
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == "ui" and isinstance(child.ctx, ast.Load):
                assert ui_bound, "dashboard/app.py uses `ui` before importing dashboard.ui"
    assert ui_bound, "dashboard/app.py must import dashboard.ui as ui"
