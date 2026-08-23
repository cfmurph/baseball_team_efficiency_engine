"""Nightly ranked fantasy-card emitter for the shared artifact lake.

Locked path (architect SoT): ``fantasy/cards.jsonl`` under ``runs/{run_id}/``
and ``current/``. ``as_of_date`` and ``schema_version`` live on each JSONL
record and on ``manifest.json`` — never in the filename.

``edge.war_source`` on card records is ``bbref`` | ``approx`` (warehouse
``real`` maps to ``bbref``). Never emit ``fangraphs``.

The emitter ranks published ``player_season_metrics`` (or the in-memory
warehouse export) into start | sit | pickup | stream. It does not fork
the pipeline or recompute WAR.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

FANTASY_CARDS_RELPATH = "fantasy/cards.jsonl"
FANTASY_SCHEMA_VERSION = "1.0"
FANTASY_WAR_SOURCES = frozenset({"bbref", "approx"})
# Voided draft filename — never emit or consume.
VOID_DATED_CARDS_PREFIX = "fantasy_cards_"
RECOMMENDATION_TYPES = ("start", "sit", "pickup", "stream")
DEFAULT_TOP_N = 8
PLAYER_METRICS_RELPATHS = (
    "metrics/player_season_metrics.csv",
    "player_season_metrics.csv",
    "current/metrics/player_season_metrics.csv",
    "current/player_season_metrics.csv",
)
PLAYER_ID_KEYS = ("player_id", "bbref_id", "mlbam_id", "id")
REQUIRED_CARD_FIELDS = (
    "schema_version",
    "as_of_date",
    "season",
    "card_id",
    "recommendation_type",
    "player",
    "edge",
    "reason",
    "rank",
)
REQUIRED_EDGE_FIELDS = (
    "vs_replacement",
    "war",
    "war_source",
    "is_approx",
    "confidence",
)
REQUIRED_PLAYER_NAME_KEYS = ("name", "player_name", "name_full")
SHARE_HEADLINES = {
    "start": "Lock this one in",
    "sit": "Park this name",
    "pickup": "Grab him",
    "stream": "Stream this arm",
}
REASON_VERBS = {
    "start": "lock this {pos} in",
    "sit": "park this {pos}",
    "pickup": "grab this {pos} before the wire dries up",
    "stream": "stream this {pos}",
}

_PITCHER_TYPES = frozenset({"pitcher", "both", "p", "sp", "rp"})
_PITCHER_POSITIONS = frozenset({"p", "sp", "rp"})


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
    """Return JSONL text. Empty / omitted records yield an empty stub file."""
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


def find_player_season_metrics(local_dir: str | Path) -> Path | None:
    """Locate published ``player_season_metrics.csv`` (lake or flat artifacts/)."""
    root = Path(local_dir)
    for rel in PLAYER_METRICS_RELPATHS:
        path = root / rel
        if path.is_file():
            return path
    return None


def default_top_n(*, environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = str(env.get("FANTASY_CARDS_TOP_N", "") or "").strip()
    if not raw:
        return DEFAULT_TOP_N
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_TOP_N
    return max(1, value)


def _cell(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            value = row[key]
            if isinstance(value, float) and pd.isna(value):
                continue
            return value
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _player_id(row: Mapping[str, Any]) -> str:
    for key in PLAYER_ID_KEYS:
        value = _cell(row, key)
        if value is not None:
            return str(value).strip()
    return ""


def _player_name(row: Mapping[str, Any]) -> str:
    value = _cell(row, "player_name", "name_full", "name", "name_last")
    return str(value).strip() if value is not None else ""


def _player_team(row: Mapping[str, Any]) -> str:
    team_id = _cell(row, "team_id")
    if team_id is not None:
        text = str(team_id).strip()
        if 2 <= len(text) <= 3:
            return text.upper()
    value = _cell(row, "team", "team_name")
    return str(value).strip() if value is not None else ""


def _player_position(row: Mapping[str, Any]) -> str:
    value = _cell(row, "position")
    if value is not None and str(value).strip():
        return str(value).strip().upper()
    player_type = str(_cell(row, "player_type") or "").strip().lower()
    if player_type in {"pitcher", "p", "sp", "rp"}:
        return "P"
    if player_type == "both":
        return "UTIL"
    if player_type == "batter":
        return "UTIL"
    return "UTIL"


def _is_pitcher(row: Mapping[str, Any]) -> bool:
    position = str(_cell(row, "position") or "").strip().lower()
    if position in _PITCHER_POSITIONS:
        return True
    player_type = str(_cell(row, "player_type") or "").strip().lower()
    return player_type in _PITCHER_TYPES


def _season(row: Mapping[str, Any]) -> int:
    return _as_int(_cell(row, "season", "year_id", "season_key"))


def _confidence(source: str, rank: int) -> float:
    base = 0.86 if source == "bbref" else 0.58
    return round(max(0.35, min(0.95, base - 0.015 * (rank - 1))), 3)


def _reason(rec_type: str, name: str, vs_replacement: float, position: str) -> str:
    verb = REASON_VERBS[rec_type].format(pos=position or "name")
    return _one_line(f"{name} is {vs_replacement:+.1f} vs replacement — {verb}.")


def _share_stat_line(
    vs_replacement: float,
    confidence: float,
    *,
    is_approx: bool,
) -> str:
    """BenchOrStart face copy. Never ``vs repl``.

    bbref: ``+X.X edge · NN% conf``. approx: ``+X.X edge`` (no confidence).
    """
    edge = f"{vs_replacement:+.1f} edge"
    if is_approx:
        return edge
    pct = confidence * 100 if confidence <= 1.0 else confidence
    return f"{edge} · {int(round(pct))}% conf"


def load_player_season_metrics(local_dir: str | Path) -> pd.DataFrame:
    """Read published player-season metrics. Empty frame when the CSV is missing."""
    path = find_player_season_metrics(local_dir)
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame()


def _latest_season_frame(player_df: pd.DataFrame) -> pd.DataFrame:
    if player_df.empty:
        return player_df
    out = player_df.copy()
    if "season" not in out.columns:
        if "year_id" in out.columns:
            out["season"] = out["year_id"]
        elif "season_key" in out.columns:
            out["season"] = out["season_key"]
    if "season" not in out.columns:
        return out
    season = pd.to_numeric(out["season"], errors="coerce")
    if season.notna().any():
        latest = season.max()
        return out.loc[season == latest].copy()
    return out


def _eligible_rows(player_df: pd.DataFrame) -> list[dict[str, Any]]:
    frame = _latest_season_frame(player_df)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        player_id = _player_id(raw)
        name = _player_name(raw)
        if not player_id or not name:
            continue
        vs_replacement = _as_float(
            _cell(raw, "vs_replacement", "war", "player_war"),
            default=float("nan"),
        )
        war = _as_float(_cell(raw, "war", "player_war", "vs_replacement"), default=float("nan"))
        if pd.isna(vs_replacement) and pd.isna(war):
            continue
        if pd.isna(vs_replacement):
            vs_replacement = war
        if pd.isna(war):
            war = vs_replacement
        raw["_vs_replacement"] = vs_replacement
        raw["_war"] = war
        raw["_surplus"] = _as_float(_cell(raw, "surplus_value", "edge"))
        raw["_pitching_war"] = _as_float(_cell(raw, "pitching_war"), default=war)
        raw["_player_id"] = player_id
        raw["_name"] = name
        rows.append(raw)
    return rows


def _sort_key(rec_type: str, row: Mapping[str, Any]) -> tuple[float, float, str]:
    vs = float(row["_vs_replacement"])
    war = float(row["_war"])
    surplus = float(row["_surplus"])
    pitching = float(row["_pitching_war"])
    player_id = str(row["_player_id"])
    if rec_type == "start":
        return (-vs, -war, player_id)
    if rec_type == "sit":
        return (vs, war, player_id)
    if rec_type == "pickup":
        return (-surplus, -vs, player_id)
    return (-pitching, -war, player_id)


def _candidates(rows: Sequence[Mapping[str, Any]], rec_type: str) -> list[Mapping[str, Any]]:
    if rec_type != "stream":
        return list(rows)
    pitchers = [row for row in rows if _is_pitcher(row)]
    return pitchers or list(rows)


def _build_card(
    row: Mapping[str, Any],
    *,
    rec_type: str,
    rank: int,
    as_of_date: str,
    schema_version: str,
) -> dict[str, object]:
    vs_replacement = round(float(row["_vs_replacement"]), 3)
    war = round(float(row["_war"]), 3)
    source = map_card_war_source(_cell(row, "war_source"))
    name = str(row["_name"])
    position = _player_position(row)
    team = _player_team(row)
    player_id = str(row["_player_id"])
    season = _season(row)
    confidence = _confidence(source, rank)
    player: dict[str, object] = {
        "player_id": player_id,
        "name": name,
        "position": position,
        "team": team,
    }
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "as_of_date": as_of_date,
        "season": season,
        "card_id": f"{as_of_date}:{rec_type}:{player_id}:{rank}",
        "recommendation_type": rec_type,
        "player": player,
        "edge": {
            "vs_replacement": vs_replacement,
            "war": war,
            "war_source": source,
            "is_approx": source == "approx",
            "confidence": confidence,
        },
        "reason": _reason(rec_type, name, vs_replacement, position),
        "rank": {"among_rec_type": rank},
        "share": {
            "headline": SHARE_HEADLINES[rec_type],
            "subtitle": " · ".join(part for part in (name, position, team) if part),
            "stat_line": _share_stat_line(
                vs_replacement, confidence, is_approx=source == "approx"
            ),
        },
    }
    return cards_record(payload, as_of_date=as_of_date, schema_version=schema_version)


def rank_fantasy_cards(
    player_df: pd.DataFrame | None,
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, object]]:
    """Rank the latest season into top-N start/sit/pickup/stream cards."""
    if player_df is None or player_df.empty:
        return []
    rows = _eligible_rows(player_df)
    if not rows:
        return []
    limit = max(1, int(top_n))
    cards: list[dict[str, object]] = []
    for rec_type in RECOMMENDATION_TYPES:
        ranked = sorted(_candidates(rows, rec_type), key=lambda row: _sort_key(rec_type, row))
        for index, row in enumerate(ranked[:limit], start=1):
            cards.append(
                _build_card(
                    row,
                    rec_type=rec_type,
                    rank=index,
                    as_of_date=as_of_date,
                    schema_version=schema_version,
                )
            )
    return cards


def card_schema_errors(card: Mapping[str, Any]) -> list[str]:
    """Return schema violations for one JSONL record. Empty means valid."""
    errors: list[str] = []
    for field in REQUIRED_CARD_FIELDS:
        if field not in card:
            errors.append(f"missing {field}")
    if str(card.get("schema_version") or "") != FANTASY_SCHEMA_VERSION:
        errors.append("schema_version must be 1.0")
    if not str(card.get("as_of_date") or "").strip():
        errors.append("as_of_date required")
    rec = str(card.get("recommendation_type") or "").strip().lower()
    if rec not in RECOMMENDATION_TYPES:
        errors.append("recommendation_type must be start|sit|pickup|stream")
    if not _one_line(str(card.get("reason") or "")):
        errors.append("reason must be a non-empty one-liner")
    elif "\n" in str(card.get("reason") or ""):
        errors.append("reason must be one line")
    player = card.get("player")
    if not isinstance(player, Mapping):
        errors.append("player must be an object")
    else:
        if not any(str(player.get(key) or "").strip() for key in REQUIRED_PLAYER_NAME_KEYS):
            errors.append("player.name required")
        if not any(str(player.get(key) or "").strip() for key in PLAYER_ID_KEYS):
            errors.append("player needs at least one id")
    edge = card.get("edge")
    if not isinstance(edge, Mapping):
        errors.append("edge must be an object")
    else:
        for field in REQUIRED_EDGE_FIELDS:
            if field not in edge:
                errors.append(f"missing edge.{field}")
        source = str(edge.get("war_source") or "").strip().lower()
        if source not in FANTASY_WAR_SOURCES:
            errors.append("edge.war_source must be bbref|approx")
        if source == "fangraphs":
            errors.append("edge.war_source cannot be fangraphs")
        is_approx = edge.get("is_approx")
        if source == "approx" and is_approx is not True:
            errors.append("is_approx must be true when war_source is approx")
        if source == "bbref" and is_approx is not False:
            errors.append("is_approx must be false when war_source is bbref")
    rank = card.get("rank")
    if not isinstance(rank, Mapping) or rank.get("among_rec_type") in (None, ""):
        errors.append("rank.among_rec_type required")
    return errors


def emit_ranked_fantasy_cards(
    local_dir: str | Path,
    *,
    as_of_date: str,
    schema_version: str = FANTASY_SCHEMA_VERSION,
    top_n: int | None = None,
    player_df: pd.DataFrame | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Write ranked ``fantasy/cards.jsonl`` from metrics. Never dated filenames.

    Reads in-memory ``player_df`` when provided; otherwise loads published
    ``player_season_metrics.csv``. Missing metrics yield an empty valid stub.
    """
    root = Path(local_dir)
    frame = player_df if player_df is not None else load_player_season_metrics(root)
    limit = default_top_n(environ=environ) if top_n is None else top_n
    records = rank_fantasy_cards(
        frame,
        as_of_date=as_of_date,
        schema_version=schema_version,
        top_n=limit,
    )
    dest = write_fantasy_cards_stub(
        root,
        as_of_date=as_of_date,
        schema_version=schema_version,
        records=records,
    )
    extra = list(root.joinpath("fantasy").glob(f"{VOID_DATED_CARDS_PREFIX}*.json"))
    extra.extend(root.glob(f"{VOID_DATED_CARDS_PREFIX}*.json"))
    for path in extra:
        path.unlink(missing_ok=True)
    return dest
