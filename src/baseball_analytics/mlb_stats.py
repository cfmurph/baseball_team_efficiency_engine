"""MLB Stats API client, raw landing, parsers, and Lahman joins.

Public majors feeds only (``sportId=1``). No API key. Does **not** ingest or
overwrite Baseball-Reference rWAR.

HTTP goes through ``python-mlb-statsapi`` (``Mlb``) so timeouts, retries, and
strict HTTP live in the library. Callers still receive JSON dicts for the
locked raw landing. Pydantic dumps are snake_case; parsers also accept the
legacy camelCase Stats API shape already on disk.

Locked raw path (ADR 0003 / #108)::

    {ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/…json
    or local data/raw/mlb_stats/{endpoint}/{as_of_date}/…json
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from mlbstatsapi import (
    Mlb,
    MlbDecodeError,
    MlbHttpError,
    MlbTimeoutError,
    MlbTransportError,
    TheMlbStatsApiException,
)

from src.baseball_analytics.storage import ArtifactBackend, default_as_of_date, open_backend

log = logging.getLogger(__name__)

STATS_API_BASE = "https://statsapi.mlb.com"
SPORT_ID = 1
LEAGUE_IDS = "103,104"  # AL, NL
RAW_REMOTE_PREFIX = "raw/mlb_stats"
RAW_LOCAL_NAME = "mlb_stats"
DEFAULT_MIN_INTERVAL_S = 0.35
DEFAULT_MAX_RETRIES = 3
# Library default is (3.05, 30.0). A scalar here is the read timeout.
DEFAULT_TIMEOUT_S = 30
DEFAULT_TEAM_MAP = "data/crosswalks/mlb_team_map.csv"
PEOPLE_MLB_ID_COLUMNS = ("mlbID", "mlb_id", "key_mlbam", "mlbam")

# Endpoint tokens used in the locked {endpoint} path segment.
ENDPOINT_TEAMS = "teams"
ENDPOINT_STANDINGS = "standings"
ENDPOINT_TEAM_HITTING = "team_hitting"
ENDPOINT_TEAM_PITCHING = "team_pitching"
ENDPOINT_PLAYER_HITTING = "player_hitting"
ENDPOINT_PLAYER_PITCHING = "player_pitching"
ENDPOINT_SCHEDULE = "schedule"
ENDPOINT_EXTRACT_REPORT = "extract_report"

SEASON_ENDPOINTS = (
    ENDPOINT_STANDINGS,
    ENDPOINT_TEAM_HITTING,
    ENDPOINT_TEAM_PITCHING,
    ENDPOINT_PLAYER_HITTING,
    ENDPOINT_PLAYER_PITCHING,
    ENDPOINT_SCHEDULE,
)

Fetcher = Callable[[str, Mapping[str, Any]], dict[str, Any]]


class MlbStatsError(RuntimeError):
    """Raised for HTTP / transport failures against the Stats API."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str = "") -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(message)


@dataclass
class EndpointResult:
    endpoint: str
    ok: bool
    relative_key: str = ""
    local_path: str = ""
    bytes_written: int = 0
    error: str | None = None
    season: int | None = None


@dataclass
class ExtractReport:
    as_of_date: str
    seasons: list[int]
    soft_fail: bool = True
    ok: bool = True
    endpoints: list[EndpointResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "seasons": list(self.seasons),
            "soft_fail": self.soft_fail,
            "ok": self.ok,
            "error": self.error,
            "endpoints": [
                {
                    "endpoint": item.endpoint,
                    "ok": item.ok,
                    "relative_key": item.relative_key,
                    "local_path": item.local_path,
                    "bytes_written": item.bytes_written,
                    "error": item.error,
                    "season": item.season,
                }
                for item in self.endpoints
            ],
        }


@dataclass
class MlbFrames:
    """Parsed warehouse-ready frames. Empty when Stats API raw is missing."""

    as_of_date: str | None = None
    team_map: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_season: pd.DataFrame = field(default_factory=pd.DataFrame)
    games: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_map: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return (
            self.team_season.empty
            and self.player_season.empty
            and self.games.empty
            and self.team_map.empty
        )


def raw_object_key(endpoint: str, as_of_date: str, filename: str) -> str:
    """Return the locked lake key ``raw/mlb_stats/{endpoint}/{as_of_date}/{file}``."""
    token = _endpoint_token(endpoint)
    date = _iso_date_token(as_of_date)
    name = Path(str(filename)).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid raw filename: {filename!r}")
    return f"{RAW_REMOTE_PREFIX}/{token}/{date}/{name}"


def local_raw_path(raw_dir: str | Path, endpoint: str, as_of_date: str, filename: str) -> Path:
    """Return ``{raw_dir}/mlb_stats/{endpoint}/{as_of_date}/{file}``."""
    token = _endpoint_token(endpoint)
    date = _iso_date_token(as_of_date)
    name = Path(str(filename)).name
    return Path(raw_dir) / RAW_LOCAL_NAME / token / date / name


def write_raw_payload(
    payload: Mapping[str, Any] | Sequence[Any],
    *,
    endpoint: str,
    as_of_date: str,
    filename: str,
    raw_dir: str | Path,
    backend: ArtifactBackend | None = None,
) -> tuple[Path, str]:
    """Write one JSON payload locally and, when configured, to ``ARTIFACTS_URI``."""
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    dest = local_raw_path(raw_dir, endpoint, as_of_date, filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    key = raw_object_key(endpoint, as_of_date, filename)
    if backend is not None:
        backend.put(key, data)
    return dest, key


def read_raw_payload(
    *,
    endpoint: str,
    as_of_date: str,
    filename: str,
    raw_dir: str | Path,
    backend: ArtifactBackend | None = None,
) -> dict[str, Any] | None:
    """Load a landed JSON object from local raw, then the shared lake."""
    dest = local_raw_path(raw_dir, endpoint, as_of_date, filename)
    if dest.is_file():
        return json.loads(dest.read_text(encoding="utf-8"))
    if backend is not None:
        data = backend.get(raw_object_key(endpoint, as_of_date, filename))
        if data is not None:
            return json.loads(data.decode("utf-8"))
    return None


def discover_as_of_dates(raw_dir: str | Path) -> list[str]:
    """Return sorted as_of_date partitions under local ``mlb_stats/``."""
    root = Path(raw_dir) / RAW_LOCAL_NAME
    if not root.is_dir():
        return []
    dates: set[str] = set()
    for path in root.glob("*/*"):
        if path.is_dir() and _is_iso_date(path.name):
            dates.add(path.name)
    return sorted(dates)


def resolve_as_of_date(
    raw_dir: str | Path,
    *,
    as_of_date: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    if as_of_date:
        return _iso_date_token(as_of_date)
    env_date = default_as_of_date(environ=environ)
    local_dates = discover_as_of_dates(raw_dir)
    if env_date in local_dates:
        return env_date
    if local_dates:
        return local_dates[-1]
    return env_date


class MlbStatsClient:
    """Stats API client backed by ``mlbstatsapi.Mlb``.

    Inject ``fetcher`` (legacy path/params hook) or ``mlb`` in tests so CI
    never touches the live network. Timeouts, retries, and strict HTTP come
    from the library when this client constructs ``Mlb``.
    """

    def __init__(
        self,
        *,
        base_url: str = STATS_API_BASE,
        min_interval: float = DEFAULT_MIN_INTERVAL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int | float | tuple[float, float] = DEFAULT_TIMEOUT_S,
        fetcher: Fetcher | None = None,
        session: Any | None = None,
        mlb: Any | None = None,
        strict_http: bool = True,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.min_interval = float(min_interval)
        self.max_retries = int(max_retries)
        self.timeout = timeout
        self.strict_http = bool(strict_http)
        self._fetcher = fetcher
        self._owns_mlb = False
        self._last_request_time = 0.0
        if mlb is not None:
            self._mlb = mlb
        elif fetcher is None:
            self._mlb = Mlb(
                hostname=_hostname_from_base(self.base_url),
                timeout=timeout,
                session=session,
                strict_http=self.strict_http,
            )
            self._owns_mlb = True
        else:
            self._mlb = None

    def close(self) -> None:
        mlb = self._mlb
        if self._owns_mlb and mlb is not None:
            closer = getattr(mlb, "close", None)
            if callable(closer):
                closer()
        self._owns_mlb = False

    def __enter__(self) -> MlbStatsClient:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        query = {key: value for key, value in dict(params or {}).items() if value is not None}
        if self._fetcher is not None:
            return self._fetcher(path, query)
        raise MlbStatsError(
            "MlbStatsClient.get() is the test fetcher hook; live calls use Mlb helpers"
        )

    def teams(self) -> dict[str, Any]:
        if self._fetcher is not None:
            return self.get("/api/v1/teams", {"sportId": SPORT_ID})
        teams = self._mlb_call(self._require_mlb().get_teams, sport_id=SPORT_ID)
        return {"teams": [_model_dump(team) for team in teams or []]}

    def standings(self, season: int) -> dict[str, Any]:
        if self._fetcher is not None:
            return self.get(
                "/api/v1/standings",
                {"leagueId": LEAGUE_IDS, "season": season, "sportId": SPORT_ID},
            )
        records = self._mlb_call(
            self._require_mlb().get_standings,
            LEAGUE_IDS,
            str(season),
            sportId=SPORT_ID,
        )
        return {"records": [_model_dump(record) for record in records or []]}

    def team_stats(self, season: int, group: str) -> dict[str, Any]:
        if self._fetcher is not None:
            return self.get(
                "/api/v1/teams/stats",
                {
                    "season": season,
                    "group": group,
                    "stats": "season",
                    "sportIds": SPORT_ID,
                },
            )
        # get_team_stats is per-team. Merge majors teams into the existing
        # landed {"stats": [{"splits": [...]}]} spine. One failed team is
        # skipped; the endpoint fails only when nothing lands.
        mlb = self._require_mlb()
        teams = self._mlb_call(mlb.get_teams, sport_id=SPORT_ID)
        splits: list[dict[str, Any]] = []
        last_error: MlbStatsError | None = None
        for team in teams or []:
            team_id = _model_id(team)
            if team_id is None:
                continue
            try:
                result = self._mlb_call(
                    mlb.get_team_stats,
                    team_id,
                    ["season"],
                    [group],
                    season=season,
                )
            except MlbStatsError as exc:
                last_error = exc
                log.warning("get_team_stats(%s, %s) failed softly: %s", team_id, group, exc)
                continue
            splits.extend(_splits_from_stat_dict(result, group))
        if not splits and last_error is not None:
            raise last_error
        return {"stats": [{"splits": splits}]}

    def player_stats(self, season: int, group: str) -> dict[str, Any]:
        if self._fetcher is not None:
            return self.get(
                "/api/v1/stats",
                {
                    "stats": "season",
                    "group": group,
                    "season": season,
                    "sportId": SPORT_ID,
                    "playerPool": "all",
                    "limit": 10000,
                },
            )
        # League-wide /stats. get_player_stats is per person and does not
        # replace this majors pool call.
        result = self._mlb_call(
            self._require_mlb().get_stats,
            ["season"],
            [group],
            season=season,
            sportId=SPORT_ID,
            playerPool="all",
            limit=10000,
        )
        return {"stats": _stat_blocks_from_group(result, group)}

    def schedule(self, *, season: int | None = None, date: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"sportId": SPORT_ID, "gameTypes": "R"}
        if date:
            params["date"] = date
        if season:
            params["season"] = season
        if self._fetcher is not None:
            return self.get("/api/v1/schedule", params)
        kwargs: dict[str, Any] = {"sport_id": SPORT_ID, "gameTypes": "R"}
        if date:
            kwargs["date"] = date
        if season:
            kwargs["season"] = season
        dumped = _model_dump(self._mlb_call(self._require_mlb().get_schedule, **kwargs))
        if isinstance(dumped, Mapping):
            return dict(dumped)
        return {"dates": []}

    def _require_mlb(self) -> Any:
        if self._mlb is None:
            raise MlbStatsError("Mlb client is not configured")
        return self._mlb

    def _mlb_call(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._throttle()
        try:
            return method(*args, **kwargs)
        except MlbHttpError as exc:
            status = getattr(exc, "status_code", None)
            url = str(getattr(exc, "url", "") or "")
            raise MlbStatsError(
                f"HTTP {status} from {url or 'statsapi'}",
                status_code=status if isinstance(status, int) else None,
                url=url,
            ) from exc
        except MlbTimeoutError as exc:
            raise MlbStatsError(f"Stats API timeout: {exc}") from exc
        except MlbTransportError as exc:
            raise MlbStatsError(f"Stats API transport failed: {exc}") from exc
        except MlbDecodeError as exc:
            raise MlbStatsError(f"Invalid JSON from Stats API: {exc}") from exc
        except TheMlbStatsApiException as exc:
            raise MlbStatsError(f"Stats API request failed: {exc}") from exc

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()


def pull_majors_feeds(
    *,
    raw_dir: str | Path,
    as_of_date: str,
    seasons: Sequence[int],
    client: MlbStatsClient,
    backend: ArtifactBackend | None = None,
    schedule_mode: str = "season",
) -> ExtractReport:
    """Fetch majors team/player/game feeds and land versioned raw JSON."""
    report = ExtractReport(as_of_date=as_of_date, seasons=list(seasons))
    report.endpoints.append(
        _pull_one(
            client.teams,
            endpoint=ENDPOINT_TEAMS,
            filename="teams.json",
            raw_dir=raw_dir,
            as_of_date=as_of_date,
            backend=backend,
        )
    )
    for year in seasons:
        report.endpoints.append(
            _pull_one(
                lambda yr=year: client.standings(yr),
                endpoint=ENDPOINT_STANDINGS,
                filename=f"standings_{year}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                season=year,
            )
        )
        report.endpoints.append(
            _pull_one(
                lambda yr=year: client.team_stats(yr, "hitting"),
                endpoint=ENDPOINT_TEAM_HITTING,
                filename=f"team_hitting_{year}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                season=year,
            )
        )
        report.endpoints.append(
            _pull_one(
                lambda yr=year: client.team_stats(yr, "pitching"),
                endpoint=ENDPOINT_TEAM_PITCHING,
                filename=f"team_pitching_{year}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                season=year,
            )
        )
        report.endpoints.append(
            _pull_one(
                lambda yr=year: client.player_stats(yr, "hitting"),
                endpoint=ENDPOINT_PLAYER_HITTING,
                filename=f"player_hitting_{year}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                season=year,
            )
        )
        report.endpoints.append(
            _pull_one(
                lambda yr=year: client.player_stats(yr, "pitching"),
                endpoint=ENDPOINT_PLAYER_PITCHING,
                filename=f"player_pitching_{year}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                season=year,
            )
        )
        if schedule_mode == "date":
            report.endpoints.append(
                _pull_one(
                    lambda: client.schedule(date=as_of_date),
                    endpoint=ENDPOINT_SCHEDULE,
                    filename=f"schedule_{as_of_date}.json",
                    raw_dir=raw_dir,
                    as_of_date=as_of_date,
                    backend=backend,
                    season=year,
                )
            )
        else:
            report.endpoints.append(
                _pull_one(
                    lambda yr=year: client.schedule(season=yr),
                    endpoint=ENDPOINT_SCHEDULE,
                    filename=f"schedule_{year}.json",
                    raw_dir=raw_dir,
                    as_of_date=as_of_date,
                    backend=backend,
                    season=year,
                )
            )
    report.ok = all(item.ok for item in report.endpoints)
    write_raw_payload(
        report.to_dict(),
        endpoint=ENDPOINT_EXTRACT_REPORT,
        as_of_date=as_of_date,
        filename="extract_report.json",
        raw_dir=raw_dir,
        backend=backend,
    )
    return report


def parse_teams(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for team in payload.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        if _nested_id(team.get("sport")) not in {None, SPORT_ID}:
            continue
        league = team.get("league") if isinstance(team.get("league"), Mapping) else {}
        division = team.get("division") if isinstance(team.get("division"), Mapping) else {}
        rows.append(
            {
                "mlb_team_id": _int(team.get("id")),
                "mlb_abbr": team.get("abbreviation"),
                "mlb_name": team.get("name"),
                "mlb_team_name": _field(team, "teamName", "team_name"),
                "mlb_location": _field(team, "locationName", "location_name"),
                "league_id": _nested_id(team.get("league")),
                "league_name": league.get("name") if league else None,
                "division_id": _nested_id(team.get("division")),
                "division_name": division.get("name") if division else None,
                "active": bool(team.get("active", True)),
            }
        )
    return _drop_null_id(pd.DataFrame(rows), "mlb_team_id")


def parse_standings(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for block in payload.get("records") or []:
        if not isinstance(block, Mapping):
            continue
        for record in _field(block, "teamRecords", "team_records") or []:
            if not isinstance(record, Mapping):
                continue
            team = record.get("team") if isinstance(record.get("team"), Mapping) else {}
            rows.append(
                {
                    "mlb_team_id": _int(team.get("id")),
                    "team_name": team.get("name"),
                    "season_year": _int(record.get("season")),
                    "wins": _int(record.get("wins")),
                    "losses": _int(record.get("losses")),
                    "games": _int(_field(record, "gamesPlayed", "games_played")),
                    "runs_scored": _int(_field(record, "runsScored", "runs_scored")),
                    "runs_allowed": _int(_field(record, "runsAllowed", "runs_allowed")),
                    "run_diff": _int(_field(record, "runDifferential", "run_differential")),
                    "winning_pct": _num(_field(record, "winningPercentage", "winning_percentage")),
                    "division_rank": _field(record, "divisionRank", "division_rank"),
                    "league_rank": _field(record, "leagueRank", "league_rank"),
                }
            )
    return _drop_null_id(pd.DataFrame(rows), "mlb_team_id")


def parse_team_stats(payload: Mapping[str, Any], group: str) -> pd.DataFrame:
    prefix = "batting" if group == "hitting" else "pitching"
    rows = []
    for split in _iter_stat_splits(payload):
        team = split.get("team") if isinstance(split.get("team"), Mapping) else {}
        stat = split.get("stat") if isinstance(split.get("stat"), Mapping) else {}
        row = {
            "mlb_team_id": _int(team.get("id")),
            "team_name": team.get("name"),
            "season_year": _int(split.get("season")),
        }
        if group == "hitting":
            row.update(
                {
                    f"{prefix}_games": _int(_field(stat, "gamesPlayed", "games_played")),
                    f"{prefix}_runs": _int(stat.get("runs")),
                    f"{prefix}_hits": _int(stat.get("hits")),
                    f"{prefix}_hr": _int(_field(stat, "homeRuns", "home_runs")),
                    f"{prefix}_bb": _int(_field(stat, "baseOnBalls", "base_on_balls")),
                    f"{prefix}_so": _int(_field(stat, "strikeOuts", "strike_outs")),
                    "avg": _num(stat.get("avg")),
                    "obp": _num(stat.get("obp")),
                    "slg": _num(stat.get("slg")),
                    "ops": _num(stat.get("ops")),
                }
            )
        else:
            row.update(
                {
                    f"{prefix}_wins": _int(stat.get("wins")),
                    f"{prefix}_losses": _int(stat.get("losses")),
                    "ip": _num(_field(stat, "inningsPitched", "innings_pitched")),
                    "era": _num(stat.get("era")),
                    "whip": _num(stat.get("whip")),
                    "pitching_so": _int(_field(stat, "strikeOuts", "strike_outs")),
                    "pitching_bb": _int(_field(stat, "baseOnBalls", "base_on_balls")),
                }
            )
        rows.append(row)
    return _drop_null_id(pd.DataFrame(rows), "mlb_team_id")


def parse_player_stats(payload: Mapping[str, Any], group: str) -> pd.DataFrame:
    rows = []
    for split in _iter_stat_splits(payload):
        player = split.get("player") if isinstance(split.get("player"), Mapping) else {}
        team = split.get("team") if isinstance(split.get("team"), Mapping) else {}
        stat = split.get("stat") if isinstance(split.get("stat"), Mapping) else {}
        row = {
            "mlb_player_id": _int(player.get("id")),
            "player_name": _field(player, "fullName", "full_name"),
            "mlb_team_id": _int(team.get("id")),
            "team_name": team.get("name"),
            "season_year": _int(split.get("season")),
        }
        if group == "hitting":
            row.update(
                {
                    "games": _int(_field(stat, "gamesPlayed", "games_played")),
                    "pa": _num(_field(stat, "plateAppearances", "plate_appearances")),
                    "ab": _num(_field(stat, "atBats", "at_bats")),
                    "hits": _num(stat.get("hits")),
                    "hr": _num(_field(stat, "homeRuns", "home_runs")),
                    "bb": _num(_field(stat, "baseOnBalls", "base_on_balls")),
                    "so": _num(_field(stat, "strikeOuts", "strike_outs")),
                    "avg": _num(stat.get("avg")),
                    "obp": _num(stat.get("obp")),
                    "slg": _num(stat.get("slg")),
                    "ops": _num(stat.get("ops")),
                }
            )
        else:
            row.update(
                {
                    "pitching_games": _int(_field(stat, "gamesPlayed", "games_played")),
                    "ip": _num(_field(stat, "inningsPitched", "innings_pitched")),
                    "era": _num(stat.get("era")),
                    "whip": _num(stat.get("whip")),
                    "pitching_so": _num(_field(stat, "strikeOuts", "strike_outs")),
                    "pitching_bb": _num(_field(stat, "baseOnBalls", "base_on_balls")),
                }
            )
        rows.append(row)
    return _drop_null_id(pd.DataFrame(rows), "mlb_player_id")


def parse_schedule(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows = []
    for day in payload.get("dates") or []:
        if not isinstance(day, Mapping):
            continue
        for game in day.get("games") or []:
            if not isinstance(game, Mapping):
                continue
            sides = game.get("teams") if isinstance(game.get("teams"), Mapping) else {}
            home = sides.get("home") if isinstance(sides.get("home"), Mapping) else {}
            away = sides.get("away") if isinstance(sides.get("away"), Mapping) else {}
            status = game.get("status") if isinstance(game.get("status"), Mapping) else {}
            venue = game.get("venue") if isinstance(game.get("venue"), Mapping) else {}
            home_record = _field(home, "leagueRecord", "league_record") or {}
            away_record = _field(away, "leagueRecord", "league_record") or {}
            if not isinstance(home_record, Mapping):
                home_record = {}
            if not isinstance(away_record, Mapping):
                away_record = {}
            official = _field(game, "officialDate", "official_date")
            game_date = _field(game, "gameDate", "game_date") or ""
            rows.append(
                {
                    "game_pk": _int(_field(game, "gamePk", "game_pk")),
                    "game_date": official or str(game_date)[:10],
                    "season_year": _int(game.get("season")),
                    "status": _field(status, "detailedState", "detailed_state")
                    or _field(status, "abstractGameState", "abstract_game_state"),
                    "venue_name": venue.get("name"),
                    "home_mlb_team_id": _nested_id(home.get("team")),
                    "away_mlb_team_id": _nested_id(away.get("team")),
                    "home_score": _int(home.get("score")),
                    "away_score": _int(away.get("score")),
                    "home_wins": _int(home_record.get("wins")),
                    "home_losses": _int(home_record.get("losses")),
                    "away_wins": _int(away_record.get("wins")),
                    "away_losses": _int(away_record.get("losses")),
                }
            )
    return _drop_null_id(pd.DataFrame(rows), "game_pk")


def load_team_map(path: str | Path = DEFAULT_TEAM_MAP) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"mlb_team_id", "lahman_team_id", "year_start", "year_end"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"mlb team map missing columns: {sorted(missing)}")
    return frame


def join_mlb_team_ids(
    df: pd.DataFrame,
    team_map: pd.DataFrame,
    *,
    mlb_team_col: str = "mlb_team_id",
    year_col: str = "season_year",
    out_col: str = "lahman_team_id",
) -> pd.DataFrame:
    """Attach year-aware Lahman team IDs. Unmatched rows keep a null bridge key."""
    if df.empty:
        out = df.copy()
        if out_col not in out.columns:
            out[out_col] = pd.Series(dtype="object")
        return out
    if mlb_team_col not in df.columns:
        raise ValueError(f"DataFrame missing {mlb_team_col}")
    mapped = team_map.rename(columns={"mlb_team_id": "_map_team_id", "lahman_team_id": out_col})
    keep = ["_map_team_id", "year_start", "year_end", out_col]
    if "lahman_franch_id" in mapped.columns and "lahman_franch_id" not in df.columns:
        keep.append("lahman_franch_id")
    mapped = mapped[keep].drop_duplicates()
    merged = df.merge(mapped, left_on=mlb_team_col, right_on="_map_team_id", how="left")
    if year_col in merged.columns:
        year = pd.to_numeric(merged[year_col], errors="coerce")
        in_window = (year >= merged["year_start"]) & (year <= merged["year_end"])
        matched = in_window.fillna(False)
        merged.loc[~matched, out_col] = pd.NA
        if "lahman_franch_id" in merged.columns:
            merged.loc[~matched, "lahman_franch_id"] = pd.NA
        merged["_matched"] = matched
        merged = (
            merged.sort_values("_matched", ascending=False)
            .drop_duplicates(subset=list(df.columns), keep="first")
            .drop(columns=["_matched"])
        )
    merged = merged.drop(columns=["_map_team_id", "year_start", "year_end"], errors="ignore")
    return merged.reset_index(drop=True)


def join_mlb_player_ids(
    df: pd.DataFrame,
    people: pd.DataFrame | None,
    *,
    mlb_player_col: str = "mlb_player_id",
    out_col: str = "lahman_player_id",
) -> pd.DataFrame:
    """Attach Lahman ``playerID`` when People exposes an MLB id column."""
    out = df.copy()
    if out_col not in out.columns:
        out[out_col] = pd.NA
    if out.empty or people is None or people.empty or mlb_player_col not in out.columns:
        return out
    id_col = next((col for col in PEOPLE_MLB_ID_COLUMNS if col in people.columns), None)
    player_col = "playerID" if "playerID" in people.columns else (
        "player_id" if "player_id" in people.columns else None
    )
    if id_col is None or player_col is None:
        return out
    bridge = people[[id_col, player_col]].dropna().copy()
    bridge[id_col] = pd.to_numeric(bridge[id_col], errors="coerce")
    bridge = bridge.dropna(subset=[id_col]).drop_duplicates(id_col)
    bridge = bridge.rename(columns={id_col: "_map_player_id", player_col: out_col})
    merged = out.drop(columns=[out_col]).merge(
        bridge, left_on=mlb_player_col, right_on="_map_player_id", how="left"
    )
    return merged.drop(columns=["_map_player_id"], errors="ignore")


def load_mlb_frames(
    raw_dir: str | Path,
    *,
    as_of_date: str | None = None,
    people: pd.DataFrame | None = None,
    team_map_path: str | Path = DEFAULT_TEAM_MAP,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> MlbFrames:
    """Parse landed Stats API JSON into warehouse frames. Empty if nothing landed."""
    resolved = resolve_as_of_date(raw_dir, as_of_date=as_of_date, environ=environ)
    frames = MlbFrames(as_of_date=resolved)
    if resolved is None:
        return frames
    try:
        team_map = load_team_map(team_map_path)
    except (OSError, ValueError) as exc:
        log.warning("MLB team map unavailable (%s); continuing without Lahman team joins", exc)
        team_map = pd.DataFrame(columns=["mlb_team_id", "lahman_team_id", "year_start", "year_end"])

    teams_payload = read_raw_payload(
        endpoint=ENDPOINT_TEAMS,
        as_of_date=resolved,
        filename="teams.json",
        raw_dir=raw_dir,
        backend=backend,
    )
    if teams_payload:
        parsed_teams = parse_teams(teams_payload)
        frames.team_map = _team_dim_from_api(parsed_teams, team_map)

    team_season_parts: list[pd.DataFrame] = []
    player_hit_parts: list[pd.DataFrame] = []
    player_pit_parts: list[pd.DataFrame] = []
    game_parts: list[pd.DataFrame] = []

    years = _discover_seasons(raw_dir, resolved, backend=backend)
    for year in years:
        standings = _read_parsed(
            parse_standings,
            ENDPOINT_STANDINGS,
            f"standings_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        hitting = _read_parsed(
            lambda payload: parse_team_stats(payload, "hitting"),
            ENDPOINT_TEAM_HITTING,
            f"team_hitting_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        pitching = _read_parsed(
            lambda payload: parse_team_stats(payload, "pitching"),
            ENDPOINT_TEAM_PITCHING,
            f"team_pitching_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        combined = _combine_team_season(standings, hitting, pitching)
        if not combined.empty:
            combined["as_of_date"] = resolved
            team_season_parts.append(join_mlb_team_ids(combined, team_map))

        player_hit = _read_parsed(
            lambda payload: parse_player_stats(payload, "hitting"),
            ENDPOINT_PLAYER_HITTING,
            f"player_hitting_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        if not player_hit.empty:
            player_hit_parts.append(player_hit)
        player_pit = _read_parsed(
            lambda payload: parse_player_stats(payload, "pitching"),
            ENDPOINT_PLAYER_PITCHING,
            f"player_pitching_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        if not player_pit.empty:
            player_pit_parts.append(player_pit)

        schedule = _read_parsed(
            parse_schedule,
            ENDPOINT_SCHEDULE,
            f"schedule_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        if schedule.empty:
            schedule = _read_parsed(
                parse_schedule,
                ENDPOINT_SCHEDULE,
                f"schedule_{resolved}.json",
                raw_dir,
                resolved,
                backend,
            )
        if not schedule.empty:
            schedule["as_of_date"] = resolved
            game_parts.append(_join_game_teams(schedule, team_map))

    if team_season_parts:
        frames.team_season = pd.concat(team_season_parts, ignore_index=True)
        frames.team_season = frames.team_season.drop_duplicates(
            ["mlb_team_id", "season_year"]
        )
    players = _merge_player_seasons(player_hit_parts, player_pit_parts)
    if not players.empty:
        players["as_of_date"] = resolved
        players = join_mlb_team_ids(players, team_map)
        players = join_mlb_player_ids(players, people)
        frames.player_season = players
        frames.player_map = (
            players[["mlb_player_id", "lahman_player_id", "player_name"]]
            .dropna(subset=["mlb_player_id"])
            .drop_duplicates("mlb_player_id")
        )
    if game_parts:
        frames.games = pd.concat(game_parts, ignore_index=True).drop_duplicates("game_pk")
    return frames


def seasons_from_settings(
    settings: Mapping[str, Any] | None,
    as_of_date: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[int]:
    env = os.environ if environ is None else environ
    raw_env = (env.get("MLB_STATS_SEASONS") or "").strip()
    if raw_env:
        return sorted({int(part) for part in raw_env.split(",") if part.strip()})
    configured = (settings or {}).get("mlb_stats") or {}
    years = configured.get("seasons") or []
    if years:
        return sorted({int(year) for year in years})
    return [int(as_of_date[:4])]


def client_from_settings(settings: Mapping[str, Any] | None = None) -> MlbStatsClient:
    configured = (settings or {}).get("mlb_stats") or {}
    timeout = configured.get("timeout")
    return MlbStatsClient(
        base_url=str(configured.get("base_url") or STATS_API_BASE),
        min_interval=float(configured.get("min_request_interval") or DEFAULT_MIN_INTERVAL_S),
        max_retries=int(configured.get("max_retries") or DEFAULT_MAX_RETRIES),
        timeout=timeout if timeout is not None else DEFAULT_TIMEOUT_S,
        strict_http=bool(configured.get("strict_http", True)),
    )


def open_optional_backend(
    uri: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ArtifactBackend | None:
    if not uri:
        return None
    return open_backend(uri, environ=environ)


def _pull_one(
    fetch: Callable[[], dict[str, Any]],
    *,
    endpoint: str,
    filename: str,
    raw_dir: str | Path,
    as_of_date: str,
    backend: ArtifactBackend | None,
    season: int | None = None,
) -> EndpointResult:
    try:
        payload = fetch()
        dest, key = write_raw_payload(
            payload,
            endpoint=endpoint,
            as_of_date=as_of_date,
            filename=filename,
            raw_dir=raw_dir,
            backend=backend,
        )
        return EndpointResult(
            endpoint=endpoint,
            ok=True,
            relative_key=key,
            local_path=str(dest),
            bytes_written=dest.stat().st_size,
            season=season,
        )
    except Exception as exc:  # soft-fail a single endpoint
        log.warning("Stats API %s failed softly: %s", endpoint, exc)
        return EndpointResult(
            endpoint=endpoint,
            ok=False,
            error=str(exc),
            season=season,
        )


def _combine_team_season(
    standings: pd.DataFrame,
    hitting: pd.DataFrame,
    pitching: pd.DataFrame,
) -> pd.DataFrame:
    parts = [frame for frame in (standings, hitting, pitching) if not frame.empty]
    if not parts:
        return pd.DataFrame()
    keys = ["mlb_team_id", "season_year"]
    out = parts[0].copy()
    for extra in parts[1:]:
        drop = [col for col in extra.columns if col == "team_name" and col in out.columns]
        out = out.merge(extra.drop(columns=drop), on=keys, how="outer")
    if "team_name" not in out.columns:
        out["team_name"] = pd.NA
    return out


def _merge_player_seasons(
    hitting_parts: Sequence[pd.DataFrame],
    pitching_parts: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    hitting = pd.concat(hitting_parts, ignore_index=True) if hitting_parts else pd.DataFrame()
    pitching = pd.concat(pitching_parts, ignore_index=True) if pitching_parts else pd.DataFrame()
    keys = ["mlb_player_id", "season_year", "mlb_team_id"]
    if hitting.empty and pitching.empty:
        return pd.DataFrame()
    if hitting.empty:
        out = pitching.copy()
        out["player_type"] = "pitcher"
        return out
    if pitching.empty:
        out = hitting.copy()
        out["player_type"] = "batter"
        return out
    pit_keep = keys + [
        c for c in pitching.columns if c not in set(hitting.columns) - set(keys) or c in {
            "ip", "era", "whip", "pitching_so", "pitching_bb", "pitching_games"
        }
    ]
    pit_keep = list(dict.fromkeys(pit_keep))
    merged = hitting.merge(pitching[pit_keep], on=keys, how="outer", suffixes=("", "_pit"))
    has_bat = merged["pa"].fillna(0) > 0 if "pa" in merged.columns else False
    has_pit = merged["ip"].fillna(0) > 0 if "ip" in merged.columns else False
    merged["player_type"] = "batter"
    merged.loc[has_pit & ~has_bat, "player_type"] = "pitcher"
    merged.loc[has_pit & has_bat, "player_type"] = "both"
    if "player_name_pit" in merged.columns:
        merged["player_name"] = merged["player_name"].fillna(merged["player_name_pit"])
        merged = merged.drop(columns=["player_name_pit"])
    return merged


def _join_game_teams(games: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    home = join_mlb_team_ids(
        games.rename(columns={"home_mlb_team_id": "mlb_team_id", "season_year": "season_year"}),
        team_map,
        out_col="home_lahman_team_id",
    )
    # join_mlb_team_ids used mlb_team_id; restore the original column name.
    if "mlb_team_id" in home.columns and "home_mlb_team_id" not in home.columns:
        home = home.rename(columns={"mlb_team_id": "home_mlb_team_id"})
    away_src = home[["away_mlb_team_id", "season_year"]].rename(
        columns={"away_mlb_team_id": "mlb_team_id"}
    )
    away = join_mlb_team_ids(away_src, team_map, out_col="away_lahman_team_id")
    away = away.rename(columns={"mlb_team_id": "away_mlb_team_id"})[
        ["away_mlb_team_id", "season_year", "away_lahman_team_id"]
    ]
    merged = home.merge(away, on=["away_mlb_team_id", "season_year"], how="left")
    return merged


def _team_dim_from_api(teams: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    if teams.empty:
        return pd.DataFrame()
    year = datetime.now(timezone.utc).year
    annotated = teams.copy()
    annotated["season_year"] = year
    joined = join_mlb_team_ids(annotated, team_map)
    cols = [
        "mlb_team_id",
        "mlb_abbr",
        "mlb_name",
        "league_id",
        "lahman_team_id",
        "lahman_franch_id",
    ]
    cols = [c for c in cols if c in joined.columns]
    return joined[cols].drop_duplicates("mlb_team_id")


def _discover_seasons(
    raw_dir: str | Path,
    as_of_date: str,
    *,
    backend: ArtifactBackend | None,
) -> list[int]:
    years: set[int] = set()
    local_root = Path(raw_dir) / RAW_LOCAL_NAME
    for endpoint in SEASON_ENDPOINTS:
        folder = local_root / endpoint / as_of_date
        if not folder.is_dir():
            continue
        for path in folder.glob("*.json"):
            year = _year_from_filename(path.name)
            if year:
                years.add(year)
    report = read_raw_payload(
        endpoint=ENDPOINT_EXTRACT_REPORT,
        as_of_date=as_of_date,
        filename="extract_report.json",
        raw_dir=raw_dir,
        backend=backend,
    )
    if report:
        for year in report.get("seasons") or []:
            years.add(int(year))
        for item in report.get("endpoints") or []:
            if item.get("season"):
                years.add(int(item["season"]))
    if years:
        return sorted(years)
    candidate = int(as_of_date[:4])
    for year in (candidate, candidate - 1):
        payload = read_raw_payload(
            endpoint=ENDPOINT_STANDINGS,
            as_of_date=as_of_date,
            filename=f"standings_{year}.json",
            raw_dir=raw_dir,
            backend=backend,
        )
        if payload:
            years.add(year)
    return sorted(years) or [candidate]


def _read_parsed(
    parser: Callable[[Mapping[str, Any]], pd.DataFrame],
    endpoint: str,
    filename: str,
    raw_dir: str | Path,
    as_of_date: str,
    backend: ArtifactBackend | None,
) -> pd.DataFrame:
    payload = read_raw_payload(
        endpoint=endpoint,
        as_of_date=as_of_date,
        filename=filename,
        raw_dir=raw_dir,
        backend=backend,
    )
    if not payload:
        return pd.DataFrame()
    try:
        return parser(payload)
    except Exception as exc:
        log.warning("Failed to parse %s/%s: %s", endpoint, filename, exc)
        return pd.DataFrame()


def _year_from_filename(name: str) -> int | None:
    for part in Path(name).stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def _endpoint_token(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid Stats API endpoint token: {value!r}")
    return text


def _iso_date_token(value: str) -> str:
    text = str(value).strip()
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _is_iso_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _nested_id(value: Any) -> int | None:
    if isinstance(value, Mapping):
        return _int(value.get("id"))
    return _int(value)


def _field(mapping: Mapping[str, Any] | None, *names: str) -> Any:
    """Return the first present key so camelCase fixtures and snake_case dumps both parse."""
    if not isinstance(mapping, Mapping):
        return None
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _iter_stat_splits(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Yield split mappings from legacy ``stats`` blocks or a library group dump."""
    splits: list[Mapping[str, Any]] = []
    blocks = payload.get("stats")
    if isinstance(blocks, Mapping):
        blocks = [blocks]
    for block in blocks or []:
        if not isinstance(block, Mapping):
            continue
        for split in block.get("splits") or []:
            if isinstance(split, Mapping):
                splits.append(split)
    return splits


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json", exclude_none=True)
    if isinstance(value, Mapping):
        return {key: _model_dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_dump(item) for item in value]
    return value


def _model_id(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _int(value.get("id"))
    return _int(getattr(value, "id", None))


def _stat_blocks_from_group(stat_dict: Mapping[str, Any] | None, group: str) -> list[dict[str, Any]]:
    if not isinstance(stat_dict, Mapping):
        return []
    group_stats = stat_dict.get(group)
    if not isinstance(group_stats, Mapping):
        return []
    blocks: list[dict[str, Any]] = []
    for stat in group_stats.values():
        dumped = _model_dump(stat)
        if isinstance(dumped, Mapping):
            blocks.append(dict(dumped))
    return blocks


def _splits_from_stat_dict(stat_dict: Mapping[str, Any] | None, group: str) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    for block in _stat_blocks_from_group(stat_dict, group):
        for split in block.get("splits") or []:
            dumped = _model_dump(split)
            if isinstance(dumped, Mapping):
                splits.append(dict(dumped))
    return splits


def _hostname_from_base(base_url: str) -> str:
    parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
    host = parsed.netloc or parsed.path
    return host.split("/")[0] or "statsapi.mlb.com"


def _int(value: Any) -> int | None:
    number = _num(value)
    if number is None:
        return None
    return int(number)


def _num(value: Any) -> float | None:
    if value in (None, "", "--", ".---"):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except ValueError:
        return None


def _drop_null_id(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    return df[df[column].notna()].reset_index(drop=True)
