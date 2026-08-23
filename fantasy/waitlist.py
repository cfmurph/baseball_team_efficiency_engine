"""Email-only waitlist capture for the BenchOrStart shell.

Default sink is a local JSONL file. Marketing can hook a real list by setting
``FANTASY_WAITLIST_WEBHOOK`` to an HTTPS endpoint (Zapier, Make, Buttondown,
Mailchimp, etc.). The POST body is JSON::

    {"email": "fan@example.com", "source": "benchorstart", "created_at": "..."}

See ``docs/fantasy.md``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re

from fantasy.copy import EMAIL_ERROR

DEFAULT_WAITLIST_PATH = Path("data/waitlist/signups.jsonl")
WAITLIST_SOURCE = "benchorstart"
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class WaitlistResult:
    ok: bool
    email: str
    path: Path
    webhook: str | None
    error: str | None = None


def normalize_email(value: str | None) -> str | None:
    text = (value or "").strip().lower()
    if not text or not _EMAIL_RE.match(text):
        return None
    return text


def waitlist_path(environ: Mapping[str, str] | None = None) -> Path:
    env = os.environ if environ is None else environ
    raw = (env.get("FANTASY_WAITLIST_PATH") or "").strip()
    return Path(raw) if raw else DEFAULT_WAITLIST_PATH


def waitlist_webhook(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    raw = (env.get("FANTASY_WAITLIST_WEBHOOK") or "").strip()
    return raw or None


def append_signup(
    email: str,
    path: Path,
    *,
    created_at: str,
    source: str = WAITLIST_SOURCE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"email": email, "source": source, "created_at": created_at}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def post_signup(
    email: str,
    webhook: str,
    *,
    created_at: str,
    source: str = WAITLIST_SOURCE,
    poster=None,
) -> None:
    import requests

    send = poster if poster is not None else requests.post
    payload = {"email": email, "source": source, "created_at": created_at}
    response = send(webhook, json=payload, timeout=10)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()


def capture_signup(
    email: str | None,
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
    poster=None,
) -> WaitlistResult:
    """Validate, append JSONL, and optionally POST to the marketing webhook."""
    path = waitlist_path(environ)
    webhook = waitlist_webhook(environ)
    cleaned = normalize_email(email)
    if cleaned is None:
        return WaitlistResult(
            ok=False,
            email=(email or "").strip(),
            path=path,
            webhook=webhook,
            error=EMAIL_ERROR,
        )
    created = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        append_signup(cleaned, path, created_at=created)
        if webhook:
            post_signup(cleaned, webhook, created_at=created, poster=poster)
    except Exception as exc:
        return WaitlistResult(
            ok=False,
            email=cleaned,
            path=path,
            webhook=webhook,
            error=str(exc),
        )
    return WaitlistResult(ok=True, email=cleaned, path=path, webhook=webhook)
