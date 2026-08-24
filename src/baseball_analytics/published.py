"""Read-only views of published ``current/`` artifacts.

Used by the thin HTTP API so Next.js never talks to the lake or vendor
keys. Resolution is the same ``resolve_artifact_hit`` / ``ARTIFACTS_URI``
contract as the dashboards — no forked reader, no invented rows.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
import json
from typing import Any

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.fantasy import (
    FANTASY_CARDS_RELPATH,
    FANTASY_SCHEMA_VERSION,
    RECOMMENDATION_TYPES,
    card_schema_errors,
)
from src.baseball_analytics.sportsdataio import default_season_window
from src.baseball_analytics.storage import (
    MANIFEST_NAME,
    SourceBadge,
    artifact_source_label,
    default_as_of_date,
    resolve_artifact_hit,
)

from fantasy.cards import parse_cards_jsonl

METRICS_MANIFEST_NAME = "metrics_manifest.json"
PLAYER_METRICS_NAME = "player_season_metrics.csv"
SECRET_ENV_KEYS = (
    "SPORTSDATAIO_API_KEY",
    "SPORTRADAR_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SESSION_TOKEN",
)


def redact_secrets(text: str, environ: Mapping[str, str] | None = None) -> str:
    """Replace secret values so they never appear in responses or logs."""
    env = environ if environ is not None else {}
    out = str(text)
    for key in SECRET_ENV_KEYS:
        value = str(env.get(key) or "").strip()
        if value:
            out = out.replace(value, f"[{key}]")
    return out


def load_json_artifact(
    filename: str,
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Load a published JSON object, or ``None`` when missing / invalid."""
    hit = resolve_artifact_hit(filename, settings, backend=backend, environ=environ)
    if hit is None:
        return None
    try:
        payload = json.loads(hit.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_published_cards(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], SourceBadge]:
    """Return schema-valid cards from ``current/fantasy/cards.jsonl``.

    Never invents cards and never falls back to BenchOrStart UI stubs.
    ``share.stat_line`` is left verbatim.
    """
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    hit = resolve_artifact_hit(
        FANTASY_CARDS_RELPATH, cfg, backend=backend, environ=environ
    )
    if hit is None:
        return [], "missing"
    try:
        text = hit.path.read_text(encoding="utf-8")
    except OSError:
        return [], "missing"
    cards = [
        card
        for card in parse_cards_jsonl(text)
        if not card_schema_errors(card)
    ]
    return cards, hit.source


def card_season(card: Mapping[str, Any]) -> int | None:
    raw = card.get("season")
    if raw in (None, ""):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def filter_cards(
    cards: Sequence[Mapping[str, Any]],
    *,
    season: int | None = None,
    rec: str | None = None,
) -> list[dict[str, Any]]:
    """Filter published cards. Does not synthesize missing seasons or recs."""
    wanted_rec = str(rec).strip().lower() if rec else None
    if wanted_rec and wanted_rec not in RECOMMENDATION_TYPES:
        raise ValueError("rec must be start|sit|pickup|stream")
    out: list[dict[str, Any]] = []
    for card in cards:
        payload = dict(card)
        if season is not None and card_season(payload) != season:
            continue
        if wanted_rec and str(payload.get("recommendation_type") or "").strip().lower() != wanted_rec:
            continue
        out.append(payload)
    return out


def seasons_from_player_metrics(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[int]:
    """Years present on published ``player_season_metrics.csv``. Empty if missing."""
    hit = resolve_artifact_hit(
        PLAYER_METRICS_NAME, settings, backend=backend, environ=environ
    )
    if hit is None:
        return []
    years: set[int] = set()
    try:
        with hit.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                raw = row.get("season") or row.get("year_id") or row.get("season_key")
                if raw in (None, ""):
                    continue
                try:
                    years.add(int(float(raw)))
                except (TypeError, ValueError):
                    continue
    except OSError:
        return []
    return sorted(years)


def published_snapshot(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Honest coverage of published ``current/`` — never invents an active season."""
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    env = environ
    source = artifact_source_label(cfg, backend=backend, environ=env)
    metrics_manifest = load_json_artifact(
        METRICS_MANIFEST_NAME, cfg, backend=backend, environ=env
    )
    lake_manifest = load_json_artifact(MANIFEST_NAME, cfg, backend=backend, environ=env)
    cards, card_source = load_published_cards(cfg, backend=backend, environ=env)
    if source == "missing" and card_source != "missing":
        source = card_source

    metric_years = seasons_from_player_metrics(cfg, backend=backend, environ=env)
    card_years = sorted({year for year in (card_season(card) for card in cards) if year is not None})
    derived_present = sorted(set(metric_years) | set(card_years))

    as_of = _first_as_of(
        metrics_manifest,
        lake_manifest,
        cards,
        environ=env,
    )
    window = _int_list((metrics_manifest or {}).get("season_window")) or default_season_window(as_of)
    active = _as_int((metrics_manifest or {}).get("active_season"))
    if active is None:
        active = window[-1] if window else int(str(as_of)[:4])

    # Years that actually appear on published metrics/cards — never union
    # a manifest claim for a season we cannot serve.
    seasons_present = derived_present

    missing = active not in seasons_present
    missing_reason = None
    if missing:
        raw_reason = (metrics_manifest or {}).get("current_season_missing_reason")
        if raw_reason:
            missing_reason = str(raw_reason)
        elif (metrics_manifest or {}).get("current_season_missing") is True:
            missing_reason = "current_season_missing"
        elif not seasons_present:
            missing_reason = "metrics_empty"
        else:
            missing_reason = "active_season_absent"

    return {
        "as_of": as_of,
        "active_season": int(active),
        "current_season_missing": bool(missing),
        "current_season_missing_reason": missing_reason if missing else None,
        "season_window": [int(year) for year in window],
        "seasons_present": [int(year) for year in seasons_present],
        "source": source,
        "schema_version": FANTASY_SCHEMA_VERSION,
        "cards": cards,
    }


def _first_as_of(
    metrics_manifest: Mapping[str, Any] | None,
    lake_manifest: Mapping[str, Any] | None,
    cards: Sequence[Mapping[str, Any]],
    *,
    environ: Mapping[str, str] | None,
) -> str:
    for payload in (metrics_manifest, lake_manifest):
        if not payload:
            continue
        raw = str(payload.get("as_of_date") or payload.get("as_of") or "").strip()
        if raw:
            return raw
    for card in cards:
        raw = str(card.get("as_of_date") or "").strip()
        if raw:
            return raw
    return default_as_of_date(environ=environ)


def _as_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: object) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    years: list[int] = []
    for item in value:
        parsed = _as_int(item)
        if parsed is not None:
            years.append(parsed)
    return years
