from __future__ import annotations

import pytest

from datetime import datetime, timezone
from pathlib import Path

from fantasy.copy import EMAIL_ERROR
from fantasy.waitlist import capture_signup, normalize_email

pytestmark = pytest.mark.unit


def test_normalize_email_accepts_simple_addresses() -> None:
    assert normalize_email("  Fan@Example.COM ") == "fan@example.com"
    assert normalize_email("not-an-email") is None
    assert normalize_email("") is None


def test_capture_signup_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "waitlist" / "signups.jsonl"
    result = capture_signup(
        "fan@example.com",
        environ={"FANTASY_WAITLIST_PATH": str(path)},
        now=datetime(2026, 8, 23, 22, 0, tzinfo=timezone.utc),
    )
    assert result.ok is True
    line = path.read_text(encoding="utf-8").strip()
    assert "fan@example.com" in line
    assert "benchorstart" in line
    assert "2026-08-23T22:00:00Z" in line


def test_capture_signup_rejects_bad_email(tmp_path: Path) -> None:
    path = tmp_path / "signups.jsonl"
    result = capture_signup("nope", environ={"FANTASY_WAITLIST_PATH": str(path)})
    assert result.ok is False
    assert result.error == EMAIL_ERROR
    assert not path.exists()


def test_capture_signup_posts_webhook(tmp_path: Path) -> None:
    path = tmp_path / "signups.jsonl"
    posted: list[tuple[str, dict]] = []

    class _Resp:
        def raise_for_status(self) -> None:
            return None

    def poster(url: str, json: dict, timeout: int):
        posted.append((url, json))
        return _Resp()

    result = capture_signup(
        "fan@example.com",
        environ={
            "FANTASY_WAITLIST_PATH": str(path),
            "FANTASY_WAITLIST_WEBHOOK": "https://hooks.example/waitlist",
        },
        now=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc),
        poster=poster,
    )
    assert result.ok is True
    assert posted == [
        (
            "https://hooks.example/waitlist",
            {
                "email": "fan@example.com",
                "source": "benchorstart",
                "created_at": "2026-08-23T12:00:00Z",
            },
        )
    ]


def test_webhook_failure_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "signups.jsonl"

    def poster(url: str, json: dict, timeout: int):
        raise RuntimeError("webhook down")

    result = capture_signup(
        "fan@example.com",
        environ={
            "FANTASY_WAITLIST_PATH": str(path),
            "FANTASY_WAITLIST_WEBHOOK": "https://hooks.example/waitlist",
        },
        poster=poster,
    )
    assert result.ok is False
    assert "webhook down" in (result.error or "")
    assert path.is_file()
