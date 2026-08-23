"""Load and normalize BenchOrStart share cards from ``cards.jsonl``.

Live keys, in order, via ``resolve_artifact`` / ``ARTIFACTS_URI``::

    current/fantasy/cards.jsonl
    fantasy/cards.jsonl

``current/`` is the published pointer (same relative file under
``runs/{run_id}/fantasy/cards.jsonl``). ``fantasy/cards.jsonl`` is optional on
the shared latest/local lake. Missing files are a miss, not an error.

Player CSVs use the same #105 helpers as the FO dashboard
(``player_season_metrics.csv`` and contract exports). Dated
``fantasy_cards_{as_of_date}.json`` is not loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
from typing import Any, Mapping

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.storage import (
    artifact_source_label,
    resolve_artifact,
    resolve_named_artifacts,
)

from fantasy.copy import EARLY_MODEL_BADGE, PROMPT_LINE

CARD_LAKE_KEY = "current/fantasy/cards.jsonl"
OPTIONAL_CARD_KEY = "fantasy/cards.jsonl"
CARD_FEED_KEYS = (CARD_LAKE_KEY, OPTIONAL_CARD_KEY)
STUB_CARDS_PATH = Path(__file__).resolve().parent / "stub_cards.jsonl"
ALLOWED_WAR_SOURCES = frozenset({"bbref", "approx"})
RETIRED_CARD_NAMES = ("fantasy_cards_",)
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


def load_stub_cards(path: Path | None = None) -> list[dict[str, Any]]:
    stub = path or STUB_CARDS_PATH
    return parse_cards_jsonl(stub.read_text(encoding="utf-8"))


def card_feed_keys() -> tuple[str, ...]:
    """Live keys this shell will ask ``resolve_artifact`` for. Never dated JSON."""
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
) -> tuple[Path | None, str, str | None]:
    """Return ``(path, source, key)`` for the first present cards.jsonl."""
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    for key in CARD_FEED_KEYS:
        path = resolve_artifact(key, cfg, backend=backend, environ=environ)
        if path is not None:
            return path, artifact_source_label(cfg), key
    return None, SOURCE_MISSING, None


def load_share_cards(
    settings: ArtifactSettings | None = None,
    *,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
    stub_path: Path | None = None,
) -> CardLoad:
    """Load live cards.jsonl when present; otherwise an empty miss (no error)."""
    path, source, key = resolve_card_feed(settings, backend=backend, environ=environ)
    if path is not None:
        cards = parse_cards_jsonl(path.read_text(encoding="utf-8"))
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


def card_headline(card: Mapping[str, Any]) -> str:
    headline = _share_map(card).get("headline")
    if headline is not None and str(headline).strip():
        return str(headline).strip()
    return recommendation_label(card.get("recommendation_type"))


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


def _format_vs_replacement(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    formatted = f"{amount:g}"
    sign = "+" if amount >= 0 and not formatted.startswith("+") else ""
    return f"{sign}{formatted} vs repl"


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


def card_stat_line(card: Mapping[str, Any]) -> str:
    stat_line = _share_map(card).get("stat_line")
    if stat_line is not None and str(stat_line).strip():
        return str(stat_line).strip()
    bits: list[str] = []
    vs_line = _format_vs_replacement(_edge_map(card).get("vs_replacement"))
    if vs_line:
        bits.append(vs_line)
    # Approx / early-model rows hide the confidence overclaim.
    if not is_approx(card):
        conf_line = _format_confidence(_confidence_value(card))
        if conf_line:
            bits.append(conf_line)
    return " · ".join(bits)


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
    prompt: str = PROMPT_LINE


def present_card(card: Mapping[str, Any]) -> ShareCardView:
    rec_type = str(card.get("recommendation_type") or "").strip().lower()
    return ShareCardView(
        recommendation_type=rec_type,
        label=recommendation_label(rec_type),
        headline=card_headline(card),
        subtitle=card_subtitle(card),
        stat_line=card_stat_line(card),
        reason=card_reason(card),
        as_of_date=card_as_of(card),
        rank_line=card_rank_line(card),
        early_model=is_approx(card),
    )


def present_cards(cards: list[Mapping[str, Any]]) -> list[ShareCardView]:
    return [present_card(card) for card in cards]


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
    stat = (
        f'<div class="bos-stat">{html.escape(view.stat_line)}</div>'
        if view.stat_line
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
    return (
        f'<article class="bos-card{featured_class}" style="--bos-tone:{tone}">'
        f'<div class="bos-wordmark">BenchOrStart</div>'
        f'<div class="bos-prompt">{html.escape(view.prompt)}</div>'
        f'<div class="bos-label">{html.escape(view.label)}</div>'
        f"{badge}{rank}"
        f'<h2 class="bos-headline">{html.escape(view.headline)}</h2>'
        f'<div class="bos-sub">{html.escape(view.subtitle)}</div>'
        f"{stat}{reason}{as_of}"
        f"</article>"
    )
