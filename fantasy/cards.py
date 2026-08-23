"""Load and normalize BenchOrStart share cards (schema 1.0).

Architect path lock via ``resolve_artifact`` / ``ARTIFACTS_URI``::

    current/fantasy/cards.jsonl

That is ``fantasy/cards.jsonl`` under the published ``current/`` prefix.
``fantasy_cards_*.json`` is a fallback only if the locked file is missing.
``edge.war_source`` is ``bbref`` or ``approx`` only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.storage import (
    default_run_date,
    resolve_artifact_hit,
    resolve_named_artifacts,
)

from fantasy.copy import EARLY_MODEL_BADGE, PROMPT_LINE

# Architect SoT: fantasy/cards.jsonl under ARTIFACTS_URI current/
CARD_LAKE_KEY = "current/fantasy/cards.jsonl"
OPTIONAL_CARD_KEY = "fantasy/cards.jsonl"
CARD_FEED_KEYS = (CARD_LAKE_KEY, OPTIONAL_CARD_KEY)
DATED_CARD_PREFIX = "fantasy_cards_"
DATED_CARD_RE = re.compile(r"fantasy_cards_(\d{4}-\d{2}-\d{2})\.json$")
STUB_CARDS_PATH = Path(__file__).resolve().parent / "stub_cards.jsonl"
ALLOWED_WAR_SOURCES = frozenset({"bbref", "approx"})
PLAYER_ARTIFACTS = {
    "players": "player_season_metrics.csv",
    "top_value": "player_top_surplus_value.csv",
    "worst": "player_worst_contracts.csv",
    "dead": "player_dead_money.csv",
}

RECOMMENDATION_LABELS = {
    "start": "START",
    "sit": "BENCH",
    "pickup": "PICK UP",
    "stream": "STREAM",
}

RANK_NOUNS = {
    "start": "start",
    "sit": "bench",
    "pickup": "pickup",
    "stream": "stream",
}

LABEL_TONES = {
    "START": "#3fb950",
    "BENCH": "#f85149",
    "PICK UP": "#58a6ff",
    "STREAM": "#d29922",
}

SOURCE_STUB = "stub"
SOURCE_MISSING = "missing"
TAB_LABELS = ("START", "BENCH", "PICK UP", "STREAM")
EDGE_UNIT = "edge"


@dataclass(frozen=True)
class CardLoad:
    cards: list[dict[str, Any]]
    source: str
    key: str | None = None
    path: Path | None = None


def recommendation_label(recommendation_type: str | None) -> str:
    key = str(recommendation_type or "").strip().lower()
    if key in RECOMMENDATION_LABELS:
        return RECOMMENDATION_LABELS[key]
    text = str(recommendation_type or "").strip()
    return text.upper() if text else "START"


def parse_cards_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse one JSON object per line; skip blanks and invalid rows."""
    cards: list[dict[str, Any]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload:
            cards.append(payload)
    return cards


def parse_card_payload(text: str, *, filename: str = "") -> list[dict[str, Any]]:
    """Parse ``cards.jsonl`` or a #111 ``fantasy_cards_*.json`` document."""
    name = Path(filename).name.lower()
    if name.endswith(".jsonl"):
        return parse_cards_jsonl(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return parse_cards_jsonl(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict) and row]
    if isinstance(payload, dict) and payload:
        nested = payload.get("cards")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict) and row]
        if payload.get("recommendation_type"):
            return [payload]
    return []


def dated_card_keys(
    settings: ArtifactSettings,
    *,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> list[str]:
    """Candidate ``fantasy_cards_{date}.json`` keys, newest / configured first."""
    dates: list[str] = []
    configured = default_run_date(now=now, environ=environ)
    dates.append(configured)
    local_dates: list[str] = []
    if settings.local_dir.is_dir():
        for path in settings.local_dir.rglob("fantasy_cards_*.json"):
            match = DATED_CARD_RE.search(path.name)
            if match:
                local_dates.append(match.group(1))
    dates.extend(sorted(set(local_dates), reverse=True))
    keys: list[str] = []
    seen: set[str] = set()
    for day in dates:
        for rel in (
            f"current/fantasy/{DATED_CARD_PREFIX}{day}.json",
            f"fantasy/{DATED_CARD_PREFIX}{day}.json",
            f"{DATED_CARD_PREFIX}{day}.json",
        ):
            if rel not in seen:
                seen.add(rel)
                keys.append(rel)
    return keys


def load_stub_cards(path: Path | None = None) -> list[dict[str, Any]]:
    stub = path or STUB_CARDS_PATH
    return parse_cards_jsonl(stub.read_text(encoding="utf-8"))


def card_feed_keys() -> tuple[str, ...]:
    """Locked jsonl first. Dated ``fantasy_cards_*.json`` is fallback-only."""
    return CARD_FEED_KEYS


def resolve_player_artifacts(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path | None]:
    """Same published player CSVs the FO dashboard loads via #105 helpers."""
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    return resolve_named_artifacts(PLAYER_ARTIFACTS, cfg, backend=backend, environ=environ)


def resolve_card_feed(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
    now: datetime | None = None,
) -> tuple[Path | None, str, str | None]:
    """Return ``(path, source, key)``. SoT is ``current/fantasy/cards.jsonl`` first."""
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    for key in (*CARD_FEED_KEYS, *dated_card_keys(cfg, environ=environ, now=now)):
        exact = cfg.local_dir / key
        if exact.is_file():
            return exact, "local", key
        literal = _literal_backend_card(key, backend, cfg)
        if literal is not None:
            return literal, "remote", key
        hit = resolve_artifact_hit(key, cfg, backend=backend, environ=environ)
        if hit is not None and _path_matches_feed_key(hit.path, key):
            return hit.path, hit.source, key
    return None, SOURCE_MISSING, None


def _literal_backend_card(
    key: str,
    backend: object | None,
    settings: ArtifactSettings,
) -> Path | None:
    """Read an exact lake key from the test/remote backend (no latest/ remap)."""
    getter = getattr(backend, "get", None)
    if getter is None:
        return None
    data = getter(key)
    if not data:
        return None
    dest = settings.cache_dir / key
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def _path_matches_feed_key(path: Path, key: str) -> bool:
    """Accept a resolve hit only when it is actually this feed key.

    ``resolve_artifact_hit`` remaps ``current/fantasy/cards.jsonl`` onto
    ``fantasy/cards.jsonl``. Keep the locked key first without treating the
    optional sibling as the SoT file.
    """
    posix = path.as_posix().replace("\\", "/")
    if not (posix.endswith("/" + key) or posix.endswith(key)):
        return False
    if key != CARD_LAKE_KEY and posix.endswith("/" + CARD_LAKE_KEY):
        return False
    return True


def load_share_cards(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
    stub_path: Path | None = None,
    now: datetime | None = None,
) -> CardLoad:
    """Load live cards when present; otherwise an empty miss (UI uses stubs)."""
    path, source, key = resolve_card_feed(
        settings, backend=backend, environ=environ, now=now
    )
    if path is not None:
        cards = parse_card_payload(path.read_text(encoding="utf-8"), filename=path.name)
        if cards:
            return CardLoad(cards=cards, source=source, key=key, path=path)
    return CardLoad(cards=[], source=SOURCE_MISSING)


def _share_map(card: Mapping[str, Any]) -> Mapping[str, Any]:
    share = card.get("share")
    return share if isinstance(share, Mapping) else {}


def _player_map(card: Mapping[str, Any]) -> Mapping[str, Any]:
    player = card.get("player")
    return player if isinstance(player, Mapping) else {}


def _edge_map(card: Mapping[str, Any]) -> Mapping[str, Any]:
    edge = card.get("edge")
    return edge if isinstance(edge, Mapping) else {}


def _rank_map(card: Mapping[str, Any]) -> Mapping[str, Any]:
    rank = card.get("rank")
    return rank if isinstance(rank, Mapping) else {}


def war_source(card: Mapping[str, Any]) -> str:
    raw = str(_edge_map(card).get("war_source") or "").strip().lower()
    if raw in ALLOWED_WAR_SOURCES:
        return raw
    return ""


def is_approx(card: Mapping[str, Any]) -> bool:
    """True when ``war_source`` is approx or ``is_approx`` is set."""
    if war_source(card) == "approx":
        return True
    flag = _edge_map(card).get("is_approx")
    return flag is True or flag == "true" or flag == 1


def player_name(card: Mapping[str, Any]) -> str:
    name = _player_map(card).get("name")
    if name in (None, ""):
        return ""
    return str(name).strip()


def card_headline(card: Mapping[str, Any]) -> str:
    """share.headline when set; otherwise the player name — never the badge label."""
    headline = _share_map(card).get("headline")
    if headline is not None and str(headline).strip():
        return str(headline).strip()
    return player_name(card)


def card_subtitle(card: Mapping[str, Any]) -> str:
    subtitle = _share_map(card).get("subtitle")
    if subtitle is not None and str(subtitle).strip():
        return str(subtitle).strip()
    player = _player_map(card)
    parts = [
        str(player[key]).strip()
        for key in ("name", "position", "team")
        if player.get(key) not in (None, "")
    ]
    return " · ".join(parts)


def _format_edge(value: Any) -> str | None:
    """Plain-language face copy for ``edge.vs_replacement``. Schema is unchanged."""
    if value in (None, ""):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    formatted = f"{amount:g}"
    sign = "+" if amount >= 0 and not formatted.startswith("+") else ""
    return f"{sign}{formatted} {EDGE_UNIT}"


def _confidence_value(card: Mapping[str, Any]) -> Any:
    edge = _edge_map(card)
    if "confidence" in edge:
        return edge.get("confidence")
    return card.get("confidence")


def _format_confidence(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    pct = raw * 100 if raw <= 1.0 else raw
    return f"{int(round(pct))}% conf"


def normalize_stat_line(text: str | None) -> str:
    """Rewrite leftover ``vs repl`` copy to ``edge``. Schema fields stay unchanged.

    Prefer omitting the line if jargon still remains after rewrite.
    """
    if text in (None, ""):
        return ""
    out = str(text)
    if not re.search(r"(?i)vs\s+repl", out):
        return " ".join(out.split())
    out = re.sub(r"(?i)vs\s+replacement", EDGE_UNIT, out)
    out = re.sub(r"(?i)vs\s+repl\b", EDGE_UNIT, out)
    cleaned = " ".join(out.split())
    if re.search(r"(?i)vs\s+repl", cleaned):
        return ""
    return cleaned


def card_stat_line(card: Mapping[str, Any]) -> str:
    stat_line = _share_map(card).get("stat_line")
    if stat_line is not None and str(stat_line).strip():
        return normalize_stat_line(str(stat_line).strip())
    bits: list[str] = []
    edge_line = _format_edge(_edge_map(card).get("vs_replacement"))
    if edge_line:
        bits.append(edge_line)
    # Approx / early-model rows hide the confidence overclaim.
    if not is_approx(card):
        conf_line = _format_confidence(_confidence_value(card))
        if conf_line:
            bits.append(conf_line)
    return normalize_stat_line(" · ".join(bits))


def card_reason(card: Mapping[str, Any]) -> str:
    reason = card.get("reason")
    if reason is None:
        return ""
    return " ".join(str(reason).split())


def card_as_of(card: Mapping[str, Any]) -> str:
    value = card.get("as_of_date")
    if value in (None, ""):
        return ""
    return str(value).strip()


def card_rank_line(card: Mapping[str, Any]) -> str:
    rank = _rank_map(card).get("among_rec_type")
    if rank in (None, ""):
        return ""
    try:
        place = int(rank)
    except (TypeError, ValueError):
        return ""
    rec = str(card.get("recommendation_type") or "").strip().lower()
    noun = RANK_NOUNS.get(rec, rec or "pick")
    return f"#{place} {noun} tonight"


@dataclass(frozen=True)
class ShareCardView:
    recommendation_type: str
    label: str
    headline: str
    subtitle: str
    stat_line: str
    reason: str
    as_of_date: str
    rank_line: str
    early_model: bool
    card_id: str = ""
    prompt: str = PROMPT_LINE


def present_card(card: Mapping[str, Any]) -> ShareCardView:
    rec_type = str(card.get("recommendation_type") or "").strip().lower()
    return ShareCardView(
        recommendation_type=rec_type,
        label=recommendation_label(rec_type),
        headline=card_headline(card),
        subtitle=card_subtitle(card),
        stat_line=normalize_stat_line(card_stat_line(card)),
        reason=card_reason(card),
        as_of_date=card_as_of(card),
        rank_line=card_rank_line(card),
        early_model=is_approx(card),
        card_id=str(card.get("card_id") or "").strip(),
    )


def present_cards(cards: list[Mapping[str, Any]]) -> list[ShareCardView]:
    return [present_card(card) for card in cards]


def cards_for_label(
    views: list[ShareCardView],
    label: str | None,
) -> list[ShareCardView]:
    """Filter presented cards by badge label. ``None`` / All returns the full list."""
    if not label:
        return list(views)
    wanted = str(label).strip().upper()
    if wanted in {"ALL", "*"}:
        return list(views)
    return [view for view in views if view.label == wanted]


def share_blurb(view: ShareCardView) -> str:
    """League-chat paste: decision + player + stat line + reason + as-of."""
    identity = view.subtitle or view.headline
    if identity:
        lines = [f"{view.label} — {identity}"]
    else:
        lines = [view.label]
    if (
        view.headline
        and identity
        and view.headline != identity
        and view.headline not in identity
    ):
        lines.append(view.headline)
    stat = normalize_stat_line(view.stat_line)
    if stat:
        lines.append(stat)
    if view.reason:
        lines.append(view.reason)
    if view.as_of_date:
        lines.append(f"as of {view.as_of_date}")
    return "\n".join(lines)


def card_share_filename(view: ShareCardView, *, ext: str = "png") -> str:
    raw = view.headline or view.subtitle or view.label
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-") or "card"
    rec = (view.recommendation_type or view.label).lower().replace(" ", "-")
    return f"benchorstart-{slug}-{rec}.{ext}"


def share_card_html(view: ShareCardView, *, featured: bool = False) -> str:
    """Screenshot-ready card markup. Escapes every field."""
    tone = LABEL_TONES.get(view.label, "#58a6ff")
    badge = (
        f'<span class="bos-badge">{html.escape(EARLY_MODEL_BADGE)}</span>'
        if view.early_model
        else ""
    )
    rank = (
        f'<div class="bos-rank">{html.escape(view.rank_line)}</div>'
        if view.rank_line
        else ""
    )
    stat_line = normalize_stat_line(view.stat_line)
    stat = (
        f'<div class="bos-stat">{html.escape(stat_line)}</div>'
        if stat_line
        else ""
    )
    reason = (
        f'<p class="bos-reason">{html.escape(view.reason)}</p>'
        if view.reason
        else ""
    )
    as_of = (
        f'<div class="bos-asof">as of {html.escape(view.as_of_date)}</div>'
        if view.as_of_date
        else ""
    )
    featured_class = " bos-card-featured" if featured else ""
    headline = (
        f'<h2 class="bos-headline">{html.escape(view.headline)}</h2>'
        if view.headline
        else ""
    )
    subtitle = (
        f'<div class="bos-sub">{html.escape(view.subtitle)}</div>'
        if view.subtitle
        else ""
    )
    return (
        f'<article class="bos-card{featured_class}" style="--bos-tone:{tone}">'
        f'<div class="bos-wordmark">BenchOrStart</div>'
        f'<div class="bos-prompt">{html.escape(view.prompt)}</div>'
        f'<div class="bos-label">{html.escape(view.label)}</div>'
        f"{badge}{rank}"
        f"{headline}{subtitle}"
        f"{stat}{reason}{as_of}"
        f"</article>"
    )
