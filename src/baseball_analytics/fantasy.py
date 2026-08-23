"""Phase 0 fantasy card stub for the shared artifact lake.

Locked path (architect SoT): ``fantasy/cards.jsonl`` under ``runs/{run_id}/``
and ``current/``. ``as_of_date`` and ``schema_version`` live on each JSONL
record and on ``manifest.json`` — never in the filename.

``edge.war_source`` on card records is ``bbref`` | ``approx`` (warehouse
``real`` maps to ``bbref``). The lake stub is a small schema-valid sample
(all four rec types) until the #111 ranked emitter.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path

FANTASY_CARDS_RELPATH = "fantasy/cards.jsonl"
FANTASY_SCHEMA_VERSION = "1.0"
FANTASY_WAR_SOURCES = frozenset({"bbref", "approx"})
# Dated array filename from a voided draft — do not emit or consume.
VOID_DATED_CARDS_PREFIX = "fantasy_cards_"
RECOMMENDATION_TYPES = ("start", "sit", "pickup", "stream")

# Small schema 1.0 sample for BenchOrStart demos. Not the #111 ranked emitter.
SAMPLE_STUB_CARDS: tuple[dict[str, object], ...] = (
    {
        "card_id": "stub-start-1",
        "recommendation_type": "start",
        "season": 2026,
        "player": {
            "player_id": "judgeaa01",
            "name": "Aaron Judge",
            "position": "OF",
            "team": "NYY",
        },
        "edge": {
            "vs_replacement": 3.4,
            "war": 6.1,
            "war_source": "bbref",
            "is_approx": False,
        },
        "reason": "Lock him in. Sitting this bat is how you lose the week.",
    },
    {
        "card_id": "stub-sit-1",
        "recommendation_type": "sit",
        "season": 2026,
        "player": {
            "player_id": "solerjo01",
            "name": "Jorge Soler",
            "position": "OF",
            "team": "LAA",
        },
        "edge": {
            "vs_replacement": -0.4,
            "war": 0.2,
            "war_source": "approx",
            "is_approx": True,
        },
        "reason": "Cold bat, tough lane — park him and play the hotter outfield.",
    },
    {
        "card_id": "stub-pickup-1",
        "recommendation_type": "pickup",
        "season": 2026,
        "player": {
            "player_id": "steersp01",
            "name": "Spencer Steer",
            "position": "1B",
            "team": "CIN",
        },
        "edge": {
            "vs_replacement": 1.6,
            "war": 2.4,
            "war_source": "bbref",
            "is_approx": False,
        },
        "reason": "Quiet week on the wire — grab him before your league chat does.",
    },
    {
        "card_id": "stub-stream-1",
        "recommendation_type": "stream",
        "season": 2026,
        "player": {
            "player_id": "suarera02",
            "name": "Ranger Suárez",
            "position": "SP",
            "team": "PHI",
        },
        "edge": {
            "vs_replacement": 1.1,
            "war": 1.8,
            "war_source": "bbref",
            "is_approx": False,
        },
        "reason": "Stream him for the matchup, then cut bait after the start.",
    },
)


def map_card_war_source(value: object) -> str:
    """Map warehouse / alias values onto card ``edge.war_source``: ``bbref`` | ``approx``.

    ``fangraphs`` is never emitted (no FG WAR ingest yet) and collapses to ``approx``.
    """
    text = str(value or "").strip().lower()
    if text in {"real", "bbref"}:
        return "bbref"
    return "approx"


def cards_record(
    row: Mapping[str, object],
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
) -> dict[str, object]:
    """Build one JSONL record. ``edge.war_source`` is ``bbref`` | ``approx`` only."""
    payload = dict(row)
    payload["schema_version"] = schema_version
    payload["as_of_date"] = as_of_date
    edge = dict(payload.get("edge") or {})
    raw_source = edge.get("war_source", payload.pop("war_source", None))
    source = map_card_war_source(raw_source)
    edge["war_source"] = source
    edge["is_approx"] = source == "approx"
    if "war" in payload and "war" not in edge:
        edge["war"] = payload["war"]
    if "vs_replacement" in payload and "vs_replacement" not in edge:
        edge["vs_replacement"] = payload["vs_replacement"]
    payload["edge"] = edge
    return payload


def render_cards_jsonl(
    records: Iterable[Mapping[str, object]] | None = None,
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
) -> str:
    """Return JSONL text. ``None`` uses the schema-valid sample; ``[]`` is empty."""
    if records is None:
        records = SAMPLE_STUB_CARDS
    if not records:
        return ""
    lines = [
        json.dumps(
            cards_record(row, as_of_date=as_of_date, schema_version=schema_version),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in records
    ]
    return "\n".join(lines) + "\n"


def write_fantasy_cards_stub(
    local_dir: str | Path,
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
    records: Iterable[Mapping[str, object]] | None = None,
) -> Path:
    """Write ``fantasy/cards.jsonl`` under ``local_dir`` (sample stub by default)."""
    dest = Path(local_dir) / FANTASY_CARDS_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        render_cards_jsonl(records, as_of_date=as_of_date, schema_version=schema_version),
        encoding="utf-8",
    )
    return dest
