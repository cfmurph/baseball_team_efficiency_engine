"""FastAPI read-only surface for published ``current/`` artifacts.

v1 is health, cards, seasons, and players. No writes, no vendor keys,
no invented 2026 rows. No warehouse / lake / SDIO pulls.
"""
from __future__ import annotations

from collections.abc import Mapping
import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.fantasy import FANTASY_SCHEMA_VERSION, RECOMMENDATION_TYPES
from src.baseball_analytics.published import (
    filter_cards,
    group_public_players,
    published_snapshot,
    redact_secrets,
    resolve_published_player,
)

DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
RecType = Literal["start", "sit", "pickup", "stream"]


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    active_season: int
    current_season_missing: bool
    season_window: list[int]
    source: Literal["remote", "local", "missing"]
    seasons_present: list[int] = Field(default_factory=list)
    current_season_missing_reason: str | None = None


class SeasonsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    active_season: int
    season_window: list[int]
    seasons_present: list[int]
    current_season_missing: bool


class CardsResponse(BaseModel):
    schema_version: str
    as_of: str
    season: int | None = None
    rec: str | None = None
    current_season_missing: bool
    cards: list[dict[str, Any]]


class FieldingLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pos: str | None = None
    g: int | float | None = None
    gs: int | float | None = None
    inn: int | float | None = None
    po: int | float | None = None
    a: int | float | None = None
    e: int | float | None = None
    dp: int | float | None = None
    pb: int | float | None = None
    fpct: int | float | None = None
    ofa: int | float | None = None
    cs: int | float | None = None
    sb: int | float | None = None
    tp: int | float | None = None
    tc: int | float | None = None
    rf: int | float | None = None
    cs_pct: int | float | None = None


class PlayerSeason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    season: int
    team: str | None = None
    team_name: str | None = None
    position: str | None = None
    player_type: str | None = None
    stat_source: str | None = None
    war_source: str | None = None
    war: int | float | None = None
    games: int | float | None = None
    pa: int | float | None = None
    ab: int | float | None = None
    hits: int | float | None = None
    hr: int | float | None = None
    bb: int | float | None = None
    so: int | float | None = None
    rbi: int | float | None = None
    sb: int | float | None = None
    runs: int | float | None = None
    doubles: int | float | None = None
    triples: int | float | None = None
    ip: int | float | None = None
    gs: int | float | None = None
    w: int | float | None = None
    l: int | float | None = None
    sv: int | float | None = None
    er: int | float | None = None
    pitching_so: int | float | None = None
    pitching_bb: int | float | None = None
    avg: int | float | None = None
    obp: int | float | None = None
    slg: int | float | None = None
    ops: int | float | None = None
    woba: int | float | None = None
    era: int | float | None = None
    whip: int | float | None = None
    fip: int | float | None = None
    putouts: int | float | None = None
    assists: int | float | None = None
    errors: int | float | None = None
    double_plays: int | float | None = None
    passed_balls: int | float | None = None
    fielding_g: int | float | None = None
    fielding_gs: int | float | None = None
    fielding_inn: int | float | None = None
    fielding_pos: str | None = None
    fpct: int | float | None = None
    cs: int | float | None = None
    hbp: int | float | None = None
    sh: int | float | None = None
    sf: int | float | None = None
    gidp: int | float | None = None
    ibb: int | float | None = None
    lob: int | float | None = None
    roe: int | float | None = None
    gsh: int | float | None = None
    singles: int | float | None = None
    tb: int | float | None = None
    xbh: int | float | None = None
    go: int | float | None = None
    ao: int | float | None = None
    ofa: int | float | None = None
    fielding_cs: int | float | None = None
    fielding_sb: int | float | None = None
    tp: int | float | None = None
    tc: int | float | None = None
    rf: int | float | None = None
    pitching_hits: int | float | None = None
    pitching_hr: int | float | None = None
    pitching_r: int | float | None = None
    cg: int | float | None = None
    sho: int | float | None = None
    hld: int | float | None = None
    bs: int | float | None = None
    svo: int | float | None = None
    qs: int | float | None = None
    gf: int | float | None = None
    bk: int | float | None = None
    wp: int | float | None = None
    np: int | float | None = None
    pk: int | float | None = None
    ir: int | float | None = None
    uer: int | float | None = None
    bf: int | float | None = None
    pitching_go: int | float | None = None
    pitching_ao: int | float | None = None
    pitching_hbp: int | float | None = None
    pitching_ibb: int | float | None = None
    iso: int | float | None = None
    babip: int | float | None = None
    sb_pct: int | float | None = None
    go_ao: int | float | None = None
    k_pct: int | float | None = None
    bb_pct: int | float | None = None
    wpct: int | float | None = None
    sv_pct: int | float | None = None
    pitching_go_ao: int | float | None = None
    k9: int | float | None = None
    bb9: int | float | None = None
    h9: int | float | None = None
    hr9: int | float | None = None
    k_bb: int | float | None = None
    pitching_k_pct: int | float | None = None
    pitching_bb_pct: int | float | None = None
    i_gs: int | float | None = None
    cs_pct: int | float | None = None
    fielding: list[FieldingLine] = Field(default_factory=list)


class PlayerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str
    name: str | None = None
    position: str | None = None
    team: str | None = None
    seasons: list[PlayerSeason] = Field(default_factory=list)


class PlayersResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    active_season: int
    current_season_missing: bool
    season_window: list[int]
    source: Literal["remote", "local", "missing"]
    seasons_present: list[int] = Field(default_factory=list)
    current_season_missing_reason: str | None = None
    season: int | None = None
    players: list[PlayerRecord] = Field(default_factory=list)


class PlayerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: str
    active_season: int
    current_season_missing: bool
    season_window: list[int]
    source: Literal["remote", "local", "missing"]
    seasons_present: list[int] = Field(default_factory=list)
    current_season_missing_reason: str | None = None
    season: int | None = None
    player: PlayerRecord | None = None


def cors_origins(environ: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    raw = str(env.get("API_CORS_ORIGINS") or "").strip()
    if raw in {"*", "all"}:
        return ["*"]
    if raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return list(DEFAULT_CORS_ORIGINS)


def cors_origin_regex(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    raw = str(env.get("API_CORS_ORIGIN_REGEX") or "").strip()
    return raw or None


def create_app(
    *,
    settings: ArtifactSettings | None = None,
    backend: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    env = dict(os.environ if environ is None else environ)
    cfg = settings if settings is not None else load_artifact_settings(environ=env)
    app = FastAPI(
        title="BTEE read API",
        version="1.0.0",
        description=(
            "Thin read-only HTTP API over published `current/` artifacts. "
            "Next.js consumes this surface and never touches the lake or "
            "SPORTSDATAIO_API_KEY. Player grain is published "
            "player_season_metrics only."
        ),
        openapi_tags=[
            {"name": "v1", "description": "Locked launch endpoints (#106)."},
        ],
    )
    app.state.settings = cfg
    app.state.backend = backend
    app.state.environ = env
    _install_secret_log_filter(env)

    origins = cors_origins(env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_origin_regex=cors_origin_regex(env),
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )

    def _snapshot() -> dict[str, Any]:
        return published_snapshot(
            app.state.settings,
            backend=app.state.backend,
            environ=app.state.environ,
        )

    @app.get("/v1/health", response_model=HealthResponse, tags=["v1"])
    def health() -> HealthResponse:
        snap = _snapshot()
        return HealthResponse(
            as_of=snap["as_of"],
            active_season=snap["active_season"],
            current_season_missing=snap["current_season_missing"],
            season_window=snap["season_window"],
            source=snap["source"],
            seasons_present=snap["seasons_present"],
            current_season_missing_reason=snap["current_season_missing_reason"],
        )

    @app.get("/v1/seasons", response_model=SeasonsResponse, tags=["v1"])
    def seasons() -> SeasonsResponse:
        snap = _snapshot()
        return SeasonsResponse(
            as_of=snap["as_of"],
            active_season=snap["active_season"],
            season_window=snap["season_window"],
            seasons_present=snap["seasons_present"],
            current_season_missing=snap["current_season_missing"],
        )

    @app.get("/v1/cards", response_model=CardsResponse, tags=["v1"])
    def cards(
        season: int | None = Query(default=None, ge=1871, le=2100),
        rec: RecType | None = Query(default=None),
    ) -> CardsResponse:
        snap = _snapshot()
        if rec is not None and rec not in RECOMMENDATION_TYPES:
            raise HTTPException(status_code=400, detail="rec must be start|sit|pickup|stream")
        try:
            filtered = filter_cards(snap["cards"], season=season, rec=rec)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CardsResponse(
            schema_version=FANTASY_SCHEMA_VERSION,
            as_of=snap["as_of"],
            season=season,
            rec=rec,
            current_season_missing=snap["current_season_missing"],
            cards=filtered,
        )

    def _honesty(snap: dict[str, Any]) -> dict[str, Any]:
        return {
            "as_of": snap["as_of"],
            "active_season": snap["active_season"],
            "current_season_missing": snap["current_season_missing"],
            "season_window": snap["season_window"],
            "source": snap["source"],
            "seasons_present": snap["seasons_present"],
            "current_season_missing_reason": snap["current_season_missing_reason"],
        }

    @app.get("/v1/players", response_model=PlayersResponse, tags=["v1"])
    def players(
        season: int | None = Query(default=None, ge=1871, le=2100),
    ) -> PlayersResponse:
        snap = _snapshot()
        window = None if season is not None else snap["season_window"]
        grouped = group_public_players(
            snap["player_seasons"],
            season=season,
            window=window,
        )
        return PlayersResponse(
            **_honesty(snap),
            season=season,
            players=[PlayerRecord.model_validate(item) for item in grouped],
        )

    @app.get("/v1/players/{id}", response_model=PlayerResponse, tags=["v1"])
    def player(
        id: str = Path(..., min_length=1, description="Internal player_id PK"),
        season: int | None = Query(default=None, ge=1871, le=2100),
    ) -> PlayerResponse:
        snap = _snapshot()
        window = None if season is not None else snap["season_window"]
        resolved = resolve_published_player(
            snap["player_seasons"],
            id,
            season=season,
            window=window,
        )
        if resolved is None:
            raise HTTPException(status_code=404, detail="player not found")
        return PlayerResponse(
            **_honesty(snap),
            season=season,
            player=PlayerRecord.model_validate(resolved),
        )

    return app


def _install_secret_log_filter(environ: Mapping[str, str]) -> None:
    class _RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.msg = redact_secrets(record.getMessage(), environ)
            record.args = ()
            return True

    logging.getLogger().addFilter(_RedactFilter())


app = create_app()
