"""Product locks for the local BenchOrStart mock session (#159)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def test_home_does_not_use_waitlist_as_cta() -> None:
    home = (WEB / "components" / "Home.tsx").read_text(encoding="utf-8")
    assert "WaitlistForm" not in home
    assert "#waitlist" not in home
    assert "CTA" not in home


def test_header_uses_local_mock_demo_user() -> None:
    header = (WEB / "components" / "SiteHeader.tsx").read_text(encoding="utf-8")
    session = (WEB / "lib" / "mock-session.ts").read_text(encoding="utf-8")
    combined = f"{header}\n{session}"
    assert "demo@benchorstart.local" in combined
    assert "Log out" in header
    assert "Log in" in header
    for banned in ("clerk", "auth.js", "next-auth", "magic link"):
        assert banned not in combined.lower()


def test_web_readme_notes_mock_and_real_login_issue() -> None:
    readme = (WEB / "README.md").read_text(encoding="utf-8")
    assert "local mock" in readme.lower() or "mock session" in readme.lower()
    assert "#158" in readme
    assert "Clerk" in readme
