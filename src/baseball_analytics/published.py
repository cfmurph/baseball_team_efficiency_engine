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


# Public player-page fields only. Cards keep share.stat_line / vs repl.
# Payroll and WAR-on-spine are never invented here.
_PLAYER_PRIVATE_KEYS = frozenset(
    {
        "salary",
        "surplus_value",
        "cost_per_war",
        "contract_label",
        "vs_replacement",
        "edge",
        "rank_overall",
        "rank_at_position",
    }
)
_PLAYER_PRIVATE_TOKENS = (
    "dfs",
    "salary",
    "betting",
    "weather",
    "box_score",
    "boxscore",
    "vs_repl",
    "vs_replacement",
)
_PLAYER_COUNTING_ALIASES = {
    "games": ("games", "g"),
    "pa": ("pa",),
    "ab": ("ab",),
    "hits": ("hits", "h"),
    "hr": ("hr",),
    "bb": ("bb",),
    "so": ("so",),
    "rbi": ("rbi",),
    "sb": ("sb",),
    "ip": ("ip",),
    "pitching_so": ("pitching_so",),
    "pitching_bb": ("pitching_bb",),
}
_PLAYER_RATE_KEYS = ("avg", "obp", "slg", "ops", "woba", "era", "whip", "fip")


def load_player_season_rows(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    """Published ``player_season_metrics.csv`` rows. Empty if the file is missing."""
    hit = resolve_artifact_hit(
        PLAYER_METRICS_NAME, settings, backend=backend, environ=environ
    )
    if hit is None:
        return []
    try:
        with hit.path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def seasons_from_player_metrics(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[int]:
    """Years present on published ``player_season_metrics.csv``. Empty if missing."""
    records = (
        list(rows)
        if rows is not None
        else load_player_season_rows(settings, backend=backend, environ=environ)
    )
    years: set[int] = set()
    for row in records:
        parsed = player_season_year(row)
        if parsed is not None:
            years.add(parsed)
    return sorted(years)


def player_season_year(row: Mapping[str, Any]) -> int | None:
    raw = row.get("season") or row.get("year_id") or row.get("season_key")
    if raw in (None, ""):
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def player_row_id(row: Mapping[str, Any]) -> str:
    return str(row.get("player_id") or "").strip()


def public_player_season(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project one published metric row. Never invents WAR, payroll, or vs repl."""
    year = player_season_year(row)
    if year is None:
        return None
    hits = _first_number(row, _PLAYER_COUNTING_ALIASES["hits"])
    ab = _first_number(row, _PLAYER_COUNTING_ALIASES["ab"])
    avg = _as_number(row.get("avg"))
    if avg is None and hits is not None and ab not in (None, 0):
        avg = round(hits / ab, 3)
    season: dict[str, Any] = {
        "season": year,
        "team": _first_text(row, ("team", "team_id")),
        "team_name": _first_text(row, ("team_name", "team")),
        "position": _first_text(row, ("position",)),
        "player_type": _first_text(row, ("player_type",)),
        "stat_source": _first_text(row, ("stat_source",)),
        "war_source": _first_text(row, ("war_source",)),
        "war": _as_number(row.get("player_war") if row.get("player_war") not in (None, "") else row.get("war")),
        "avg": _json_number(avg),
    }
    for public, aliases in _PLAYER_COUNTING_ALIASES.items():
        season[public] = _json_number(_first_number(row, aliases))
    for key in _PLAYER_RATE_KEYS:
        if key == "avg":
            continue
        season[key] = _json_number(_as_number(row.get(key)))
    return season


def public_player_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "player_id": player_row_id(row),
        "name": _first_text(row, ("player_name", "name_full", "name", "display_name")),
        "position": _first_text(row, ("position",)),
        "team": _first_text(row, ("team", "team_id")),
    }


def group_public_players(
    rows: Sequence[Mapping[str, Any]],
    *,
    season: int | None = None,
    window: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Group published rows by internal ``player_id``. Empty seasons are omitted."""
    wanted = int(season) if season is not None else None
    allowed = {int(year) for year in window} if window is not None else None
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        if _row_is_private_only(row):
            continue
        player_id = player_row_id(row)
        year = player_season_year(row)
        if not player_id or year is None:
            continue
        if wanted is not None and year != wanted:
            continue
        if allowed is not None and year not in allowed:
            continue
        public = public_player_season(row)
        if public is None:
            continue
        bucket = grouped.get(player_id)
        if bucket is None:
            identity = public_player_identity(row)
            bucket = {**identity, "seasons": []}
            grouped[player_id] = bucket
            order.append(player_id)
        else:
            identity = public_player_identity(row)
            if not bucket.get("name"):
                bucket["name"] = identity["name"]
            if public["season"] >= max((item["season"] for item in bucket["seasons"]), default=-1):
                bucket["position"] = identity["position"] or bucket.get("position")
                bucket["team"] = identity["team"] or bucket.get("team")
        bucket["seasons"].append(public)
    players: list[dict[str, Any]] = []
    for player_id in order:
        payload = grouped[player_id]
        payload["seasons"] = sorted(
            payload["seasons"], key=lambda item: int(item["season"]), reverse=True
        )
        if payload["seasons"]:
            players.append(payload)
    players.sort(key=lambda item: (str(item.get("name") or "").lower(), item["player_id"]))
    return players


def resolve_published_player(
    rows: Sequence[Mapping[str, Any]],
    player_id: str,
    *,
    season: int | None = None,
    window: Sequence[int] | None = None,
) -> dict[str, Any] | None:
    """Resolve URL ``id`` as the published internal ``player_id`` PK.

    Returns identity with an empty ``seasons`` list when the player exists
    but the requested year / default window has no published row. ``None``
    when the PK is absent from published metrics.
    """
    wanted = str(player_id or "").strip()
    if not wanted:
        return None
    matches = [row for row in rows if player_row_id(row) == wanted]
    if not matches:
        return None
    grouped = group_public_players(matches, season=season, window=window)
    if grouped:
        return grouped[0]
    latest = max(matches, key=lambda row: player_season_year(row) or -1)
    identity = public_player_identity(latest)
    identity["seasons"] = []
    return identity


def _row_is_private_only(row: Mapping[str, Any]) -> bool:
    """True when a row is only private keys (should not happen on published CSV)."""
    keys = {str(key).strip().lower() for key in row if str(key).strip()}
    if not keys:
        return True
    return keys <= _PLAYER_PRIVATE_KEYS


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        raw = row.get(key)
        if raw in (None, ""):
            continue
        text = str(raw).strip()
        if text:
            return text
    return None


def _first_number(row: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        parsed = _as_number(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _as_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    if value != value:  # NaN
        return None
    if float(value).is_integer():
        return int(value)
    return float(value)


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

    player_seasons = load_player_season_rows(cfg, backend=backend, environ=env)
    metric_years = seasons_from_player_metrics(rows=player_seasons)
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
        "player_seasons": player_seasons,
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
