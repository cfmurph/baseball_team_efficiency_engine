"""Phase 0 fantasy card stub for the shared artifact lake.

Locked path (architect SoT): ``fantasy/cards.jsonl`` under ``runs/{run_id}/``
and ``current/``. ``as_of_date`` and ``schema_version`` live on each JSONL
record and on ``manifest.json`` — never in the filename.

``war_source`` on card records is ``bbref`` | ``approx`` (warehouse
``real`` maps to ``bbref``). Empty/stub files are valid until the marketing
card schema lands.
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


def map_card_war_source(value: object) -> str:
    """Map warehouse ``real|approx|mixed`` onto the card enum ``bbref|approx``."""
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
    """Build one JSONL record. Extra keys from ``row`` are kept if present."""
    payload = dict(row)
    payload["schema_version"] = schema_version
    payload["as_of_date"] = as_of_date
    payload["war_source"] = map_card_war_source(payload.get("war_source"))
    return payload


def render_cards_jsonl(
    records: Iterable[Mapping[str, object]] | None = None,
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
) -> str:
    """Return JSONL text. Empty input yields an empty (valid) stub file."""
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
    """Write ``fantasy/cards.jsonl`` under ``local_dir`` (empty stub by default)."""
    dest = Path(local_dir) / FANTASY_CARDS_RELPATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        render_cards_jsonl(records, as_of_date=as_of_date, schema_version=schema_version),
        encoding="utf-8",
    )
    return dest
