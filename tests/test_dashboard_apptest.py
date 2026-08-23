"""Streamlit AppTest: every sidebar page boots without exception."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.helpers import nav_labels

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


def test_all_sidebar_pages_boot_without_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Would have caught #103: missing _status / page_* aliases / NameError on boot."""
    monkeypatch.chdir(ROOT)
    labels = nav_labels()
    assert labels, "nav_labels() must list every sidebar page"

    at = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    _fail_if_exception(at, "initial boot")

    radio = at.sidebar.radio[0]
    assert list(radio.options) == labels

    for label in labels:
        at.sidebar.radio[0].set_value(label).run()
        _fail_if_exception(at, label)
