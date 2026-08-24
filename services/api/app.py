"""FastAPI read-only surface for published ``current/`` artifacts.

v1 is locked to health, cards, and seasons. No writes, no vendor keys,
no invented 2026 rows.
"""
from __future__ import annotations

from collections.abc import Mapping
import logging
import os
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.fantasy import FANTASY_SCHEMA_VERSION, RECOMMENDATION_TYPES
from src.baseball_analytics.published import (
    filter_cards,
    published_snapshot,
    redact_secrets,
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
            "SPORTSDATAIO_API_KEY."
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

    return app


def _install_secret_log_filter(environ: Mapping[str, str]) -> None:
    class _RedactFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            record.msg = redact_secrets(record.getMessage(), environ)
            record.args = ()
            return True

    logging.getLogger().addFilter(_RedactFilter())


app = create_app()
