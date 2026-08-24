"""SportsDataIO client, raw landing, parsers, and Phase 0 warehouse spine.

Primary live ingest for schema v0.1 (LOCKED by Cole 2026-08-23 / #128).
Reads ``SPORTSDATAIO_API_KEY`` from the environment — never hardcoded.

Locked raw path::

    {ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/…json
    or local data/raw/sportsdataio/{endpoint}/{as_of_date}/…json

Does **not** replace Lahman, Baseball-Reference rWAR, or the MLB Stats API
path. Does **not** create ``fantasy_*_stat`` / ``scout_*_stat`` tables.
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
import uuid

import pandas as pd
import requests

from src.baseball_analytics.io import DEFAULT_HEADERS
from src.baseball_analytics.storage import (
    ArtifactBackend,
    default_as_of_date,
    default_run_id,
    open_backend,
)

log = logging.getLogger(__name__)

API_KEY_ENV = "SPORTSDATAIO_API_KEY"
SDIO_API_BASE = "https://api.sportsdata.io"
SDIO_UUID_NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://api.sportsdata.io/v3/mlb")
RAW_REMOTE_PREFIX = "raw/sportsdataio"
RAW_LOCAL_NAME = "sportsdataio"
DEFAULT_MIN_INTERVAL_S = 0.5
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_S = 45
DEFAULT_TEAM_MAP = "data/crosswalks/mlb_team_map.csv"
ALIAS_SYSTEMS = ("sportsdataio", "mlb", "bbref", "fangraphs", "lahman")
ENTITY_TYPES = ("player", "team", "game")
ACCOUNT_TYPES = ("fantasy", "scout", "operator_api")
PEOPLE_MLB_ID_COLUMNS = ("mlbID", "mlb_id", "key_mlbam", "mlbam", "MLBAMID")
PEOPLE_BBREF_COLUMNS = ("bbrefID", "bbref_id", "key_bbref")
SOURCE_NAME = "sportsdataio"

ENDPOINT_TEAMS = "teams"
ENDPOINT_PLAYERS = "players"
ENDPOINT_GAMES_BY_DATE = "games_by_date"
ENDPOINT_GAMES = "games"
ENDPOINT_PLAYER_GAME_STATS = "player_game_stats"
ENDPOINT_PLAYER_SEASON_STATS = "player_season_stats"
ENDPOINT_EXTRACT_REPORT = "extract_report"

DATE_ENDPOINTS = (ENDPOINT_GAMES_BY_DATE, ENDPOINT_PLAYER_GAME_STATS)
SEASON_ENDPOINTS = (ENDPOINT_GAMES, ENDPOINT_PLAYER_SEASON_STATS)

_MONTH_ABB = (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
)

Fetcher = Callable[[str, Mapping[str, Any]], Any]


class SportsDataIOError(RuntimeError):
    """Raised for HTTP / transport failures against SportsDataIO."""

    def __init__(self, message: str, *, status_code: int | None = None, url: str = "") -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(message)


class MissingApiKeyError(SportsDataIOError):
    """Raised when ``SPORTSDATAIO_API_KEY`` is unset. Extract soft-fails."""

    def __init__(self) -> None:
        super().__init__(
            f"{API_KEY_ENV} is not set; SportsDataIO extract will soft-fail",
            status_code=None,
            url="",
        )


@dataclass
class EndpointResult:
    endpoint: str
    ok: bool
    relative_key: str = ""
    local_path: str = ""
    bytes_written: int = 0
    error: str | None = None
    season: int | None = None
    as_of_date: str | None = None


@dataclass
class ExtractReport:
    as_of_date: str
    seasons: list[int]
    soft_fail: bool = True
    ok: bool = True
    endpoints: list[EndpointResult] = field(default_factory=list)
    error: str | None = None
    skipped_reason: str | None = None
    active_season: int | None = None
    current_season_missing: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "seasons": list(self.seasons),
            "soft_fail": self.soft_fail,
            "ok": self.ok,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
            "source": SOURCE_NAME,
            "schema_version": "0.1",
            "active_season": self.active_season,
            "current_season_missing": self.current_season_missing,
            "endpoints": [
                {
                    "endpoint": item.endpoint,
                    "ok": item.ok,
                    "relative_key": item.relative_key,
                    "local_path": item.local_path,
                    "bytes_written": item.bytes_written,
                    "error": item.error,
                    "season": item.season,
                    "as_of_date": item.as_of_date,
                }
                for item in self.endpoints
            ],
        }


@dataclass
class SdioFrames:
    """Parsed warehouse-ready spine frames. Empty when SDIO raw is missing."""

    as_of_date: str | None = None
    players: pd.DataFrame = field(default_factory=pd.DataFrame)
    teams: pd.DataFrame = field(default_factory=pd.DataFrame)
    games: pd.DataFrame = field(default_factory=pd.DataFrame)
    aliases: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_game_stat: pd.DataFrame = field(default_factory=pd.DataFrame)
    player_season_stat: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def empty(self) -> bool:
        return (
            self.players.empty
            and self.teams.empty
            and self.games.empty
            and self.aliases.empty
            and self.player_game_stat.empty
            and self.player_season_stat.empty
        )


def resolve_api_key(
    api_key: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return a trimmed API key from the argument or ``SPORTSDATAIO_API_KEY``."""
    if api_key is not None and str(api_key).strip():
        return str(api_key).strip()
    env = os.environ if environ is None else environ
    raw = (env.get(API_KEY_ENV) or "").strip()
    return raw or None


def sdio_date_token(value: str) -> str:
    """Convert ``YYYY-MM-DD`` to SportsDataIO's ``YYYY-MMM-DD`` path token."""
    text = _iso_date_token(value)
    stamp = datetime.strptime(text, "%Y-%m-%d")
    return f"{stamp.year}-{_MONTH_ABB[stamp.month - 1]}-{stamp.day:02d}"


def stable_uuid(kind: str, external_id: object) -> str:
    """Deterministic UUID5 so rebuilds keep the same internal PK."""
    token = str(external_id).strip()
    if not token:
        raise ValueError(f"Cannot mint UUID for empty {kind} id")
    return str(uuid.uuid5(SDIO_UUID_NS, f"sportsdataio:{kind}:{token}"))


def raw_object_key(endpoint: str, as_of_date: str, filename: str) -> str:
    """Return the locked lake key ``raw/sportsdataio/{endpoint}/{as_of_date}/{file}``."""
    token = _endpoint_token(endpoint)
    date = _iso_date_token(as_of_date)
    name = Path(str(filename)).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid raw filename: {filename!r}")
    return f"{RAW_REMOTE_PREFIX}/{token}/{date}/{name}"


def local_raw_path(raw_dir: str | Path, endpoint: str, as_of_date: str, filename: str) -> Path:
    """Return ``{raw_dir}/sportsdataio/{endpoint}/{as_of_date}/{file}``."""
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
) -> Any | None:
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
    """Return sorted as_of_date partitions under local ``sportsdataio/``."""
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


class SportsDataIOClient:
    """Polite SDIO client. Inject ``fetcher`` in tests to avoid the network.

    The subscription key is sent only as ``Ocp-Apim-Subscription-Key``.
    It is never written into raw JSON or log lines.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SDIO_API_BASE,
        min_interval: float = DEFAULT_MIN_INTERVAL_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT_S,
        fetcher: Fetcher | None = None,
        session: requests.Session | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.api_key = resolve_api_key(api_key, environ=environ)
        self.base_url = str(base_url).rstrip("/")
        self.min_interval = float(min_interval)
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self._fetcher = fetcher
        self._session = session or requests.Session()
        self._session.headers.update(DEFAULT_HEADERS)
        self._session.headers.setdefault("Accept", "application/json")
        self._last_request_time = 0.0

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        query = {key: value for key, value in dict(params or {}).items() if value is not None}
        if self._fetcher is not None:
            return self._fetcher(path, query)
        if not self.api_key:
            raise MissingApiKeyError()
        return self._http_get(path, query)

    def teams(self) -> Any:
        return self.get("/v3/mlb/scores/json/Teams")

    def players(self) -> Any:
        return self.get("/v3/mlb/scores/json/Players")

    def games_by_date(self, date: str) -> Any:
        return self.get(f"/v3/mlb/scores/json/GamesByDate/{sdio_date_token(date)}")

    def games(self, season: int) -> Any:
        return self.get(f"/v3/mlb/scores/json/Games/{int(season)}")

    def player_game_stats_by_date(self, date: str) -> Any:
        return self.get(f"/v3/mlb/stats/json/PlayerGameStatsByDate/{sdio_date_token(date)}")

    def player_season_stats(self, season: int) -> Any:
        return self.get(f"/v3/mlb/stats/json/PlayerSeasonStats/{int(season)}")

    def _http_get(self, path: str, params: Mapping[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Ocp-Apim-Subscription-Key": self.api_key or ""}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self._session.get(
                    url, params=params, headers=headers, timeout=self.timeout
                )
            except requests.RequestException as exc:
                last_error = exc
                _backoff(attempt)
                continue
            if response.status_code in {401, 403}:
                raise SportsDataIOError(
                    f"HTTP {response.status_code} from SportsDataIO (key rejected or unauthorized)",
                    status_code=response.status_code,
                    url=path,
                )
            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = SportsDataIOError(
                    f"HTTP {response.status_code} from SportsDataIO {path}",
                    status_code=response.status_code,
                    url=path,
                )
                _backoff(attempt)
                continue
            if response.status_code >= 400:
                raise SportsDataIOError(
                    f"HTTP {response.status_code} from SportsDataIO {path}: {response.text[:200]}",
                    status_code=response.status_code,
                    url=path,
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise SportsDataIOError(f"Invalid JSON from SportsDataIO {path}") from exc
            return payload
        raise SportsDataIOError(f"SportsDataIO request failed after retries: {path}: {last_error}")

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()


def pull_phase0_feeds(
    *,
    raw_dir: str | Path,
    as_of_date: str,
    seasons: Sequence[int],
    client: SportsDataIOClient,
    backend: ArtifactBackend | None = None,
    include_season_feeds: bool = False,
    pull_dates: Sequence[str] | None = None,
) -> ExtractReport:
    """Fetch Phase 0 SDIO feeds and land versioned raw JSON.

    Default is incremental: Teams + Players bootstrap, then
    ``GamesByDate`` / ``PlayerGameStatsByDate`` for ``as_of_date``.
    Season-wide games / player-season stats are opt-in.
    """
    report = ExtractReport(as_of_date=as_of_date, seasons=list(seasons))
    if not client.api_key and client._fetcher is None:
        report.ok = False
        report.soft_fail = True
        report.error = f"missing_{API_KEY_ENV}"
        report.skipped_reason = "missing_api_key"
        mark_extract_season_coverage(report)
        _write_report(report, raw_dir, as_of_date, backend)
        log.warning("SportsDataIO extract skipped: %s unset", API_KEY_ENV)
        return report

    dates = list(pull_dates) if pull_dates else [as_of_date]

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
    report.endpoints.append(
        _pull_one(
            client.players,
            endpoint=ENDPOINT_PLAYERS,
            filename="players.json",
            raw_dir=raw_dir,
            as_of_date=as_of_date,
            backend=backend,
        )
    )
    for day in dates:
        report.endpoints.append(
            _pull_one(
                lambda d=day: client.games_by_date(d),
                endpoint=ENDPOINT_GAMES_BY_DATE,
                filename=f"games_by_date_{day}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                date=day,
            )
        )
        report.endpoints.append(
            _pull_one(
                lambda d=day: client.player_game_stats_by_date(d),
                endpoint=ENDPOINT_PLAYER_GAME_STATS,
                filename=f"player_game_stats_{day}.json",
                raw_dir=raw_dir,
                as_of_date=as_of_date,
                backend=backend,
                date=day,
            )
        )
    if include_season_feeds:
        for year in seasons:
            report.endpoints.append(
                _pull_one(
                    lambda yr=year: client.games(yr),
                    endpoint=ENDPOINT_GAMES,
                    filename=f"games_{year}.json",
                    raw_dir=raw_dir,
                    as_of_date=as_of_date,
                    backend=backend,
                    season=year,
                )
            )
            report.endpoints.append(
                _pull_one(
                    lambda yr=year: client.player_season_stats(yr),
                    endpoint=ENDPOINT_PLAYER_SEASON_STATS,
                    filename=f"player_season_stats_{year}.json",
                    raw_dir=raw_dir,
                    as_of_date=as_of_date,
                    backend=backend,
                    season=year,
                )
            )
    else:
        for year in seasons:
            report.endpoints.append(
                _pull_one(
                    lambda yr=year: client.player_season_stats(yr),
                    endpoint=ENDPOINT_PLAYER_SEASON_STATS,
                    filename=f"player_season_stats_{year}.json",
                    raw_dir=raw_dir,
                    as_of_date=as_of_date,
                    backend=backend,
                    season=year,
                )
            )
    report.ok = all(item.ok for item in report.endpoints)
    mark_extract_season_coverage(report)
    _write_report(report, raw_dir, as_of_date, backend)
    return report


def parse_teams(payload: Any) -> pd.DataFrame:
    rows = []
    for team in _as_records(payload):
        sdio_id = _int(team.get("TeamID"))
        if sdio_id is None:
            continue
        rows.append(
            {
                "sdio_team_id": sdio_id,
                "sdio_abbr": team.get("Key") or team.get("Team"),
                "city": team.get("City"),
                "team_name": team.get("Name"),
                "league": team.get("League"),
                "division": team.get("Division"),
                "active": bool(team.get("Active", True)),
                "global_team_id": _int(team.get("GlobalTeamID")),
            }
        )
    return _drop_null_id(pd.DataFrame(rows), "sdio_team_id")


def parse_players(payload: Any) -> pd.DataFrame:
    rows = []
    for player in _as_records(payload):
        sdio_id = _int(player.get("PlayerID"))
        if sdio_id is None:
            continue
        mlb_id = _int(
            player.get("MLBAMID")
            or player.get("MlbID")
            or player.get("MLBamid")
            or player.get("SportsDataMLBAMID")
        )
        first = player.get("FirstName") or ""
        last = player.get("LastName") or ""
        full = player.get("FanDuelName") or f"{first} {last}".strip()
        rows.append(
            {
                "sdio_player_id": sdio_id,
                "sdio_team_id": _int(player.get("TeamID")),
                "sdio_abbr": player.get("Team"),
                "first_name": first or None,
                "last_name": last or None,
                "display_name": full or None,
                "position": player.get("Position"),
                "bats": player.get("BatHand"),
                "throws": player.get("ThrowHand"),
                "mlb_player_id": mlb_id,
                "status": player.get("Status"),
            }
        )
    return _drop_null_id(pd.DataFrame(rows), "sdio_player_id")


def parse_games(payload: Any) -> pd.DataFrame:
    rows = []
    for game in _as_records(payload):
        sdio_id = _int(game.get("GameID"))
        if sdio_id is None:
            continue
        day = game.get("Day") or game.get("DateTime") or ""
        rows.append(
            {
                "sdio_game_id": sdio_id,
                "game_date": str(day)[:10] or None,
                "season": _int(game.get("Season")),
                "season_type": _int(game.get("SeasonType")),
                "status": game.get("Status"),
                "home_sdio_team_id": _int(game.get("HomeTeamID")),
                "away_sdio_team_id": _int(game.get("AwayTeamID")),
                "home_abbr": game.get("HomeTeam"),
                "away_abbr": game.get("AwayTeam"),
                "home_score": _int(game.get("HomeTeamRuns")),
                "away_score": _int(game.get("AwayTeamRuns")),
            }
        )
    return _drop_null_id(pd.DataFrame(rows), "sdio_game_id")


def parse_player_game_stats(payload: Any) -> pd.DataFrame:
    rows = []
    for stat in _as_records(payload):
        player_id = _int(stat.get("PlayerID"))
        game_id = _int(stat.get("GameID"))
        if player_id is None or game_id is None:
            continue
        day = stat.get("Day") or stat.get("DateTime") or ""
        rows.append(
            {
                "sdio_player_id": player_id,
                "sdio_game_id": game_id,
                "sdio_team_id": _int(stat.get("TeamID")),
                "display_name": stat.get("Name"),
                "position": stat.get("Position"),
                "game_date": str(day)[:10] or None,
                "season": _int(stat.get("Season")),
                "started": _int(stat.get("Started")),
                "games": _int(stat.get("Games")),
                "pa": _num(stat.get("PlateAppearances")),
                "ab": _num(stat.get("AtBats")),
                "runs": _num(stat.get("Runs")),
                "hits": _num(stat.get("Hits")),
                "doubles": _num(stat.get("Doubles")),
                "triples": _num(stat.get("Triples")),
                "hr": _num(stat.get("HomeRuns")),
                "rbi": _num(stat.get("RunsBattedIn")),
                "bb": _num(stat.get("Walks")),
                "so": _num(stat.get("Strikeouts")),
                "sb": _num(stat.get("StolenBases")),
                "hbp": _num(stat.get("HitByPitch")),
                "avg": _num(stat.get("BattingAverage")),
                "obp": _num(stat.get("OnBasePercentage")),
                "slg": _num(stat.get("SluggingPercentage")),
                "ops": _num(stat.get("OnBasePlusSlugging")),
                "ip": _num(stat.get("InningsPitchedDecimal")),
                "er": _num(stat.get("PitchingEarnedRuns")),
                "era": _num(stat.get("EarnedRunAverage")),
                "whip": _num(stat.get("WalksHitsPerInningsPitched")),
                "pitching_so": _num(stat.get("PitchingStrikeouts")),
                "pitching_bb": _num(stat.get("PitchingWalks")),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def parse_player_season_stats(payload: Any) -> pd.DataFrame:
    rows = []
    for stat in _as_records(payload):
        player_id = _int(stat.get("PlayerID"))
        if player_id is None:
            continue
        rows.append(
            {
                "sdio_player_id": player_id,
                "sdio_team_id": _int(stat.get("TeamID")),
                "display_name": stat.get("Name"),
                "position": stat.get("Position"),
                "season": _int(stat.get("Season")),
                "games": _int(stat.get("Games")),
                "pa": _num(stat.get("PlateAppearances")),
                "ab": _num(stat.get("AtBats")),
                "hits": _num(stat.get("Hits")),
                "hr": _num(stat.get("HomeRuns")),
                "bb": _num(stat.get("Walks")),
                "so": _num(stat.get("Strikeouts")),
                "rbi": _num(stat.get("RunsBattedIn")),
                "sb": _num(stat.get("StolenBases")),
                "ip": _num(stat.get("InningsPitchedDecimal")),
                "era": _num(stat.get("EarnedRunAverage")),
                "whip": _num(stat.get("WalksHitsPerInningsPitched")),
                "pitching_so": _num(stat.get("PitchingStrikeouts")),
                "pitching_bb": _num(stat.get("PitchingWalks")),
            }
        )
    return _drop_null_id(pd.DataFrame(rows), "sdio_player_id")


def attach_lahman_aliases(
    players: pd.DataFrame,
    people: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach ``lahman_player_id`` and ``bbref_id`` when People has an MLB id."""
    out = players.copy()
    if "lahman_player_id" not in out.columns:
        out["lahman_player_id"] = pd.NA
    if "bbref_id" not in out.columns:
        out["bbref_id"] = pd.NA
    if out.empty or people is None or people.empty or "mlb_player_id" not in out.columns:
        return out
    mlb_col = next((col for col in PEOPLE_MLB_ID_COLUMNS if col in people.columns), None)
    player_col = "playerID" if "playerID" in people.columns else (
        "player_id" if "player_id" in people.columns else None
    )
    bbref_col = next((col for col in PEOPLE_BBREF_COLUMNS if col in people.columns), None)
    if mlb_col is None or player_col is None:
        return out
    keep = [mlb_col, player_col] + ([bbref_col] if bbref_col else [])
    bridge = people[keep].dropna(subset=[mlb_col]).copy()
    bridge[mlb_col] = pd.to_numeric(bridge[mlb_col], errors="coerce")
    bridge = bridge.dropna(subset=[mlb_col]).drop_duplicates(mlb_col)
    rename = {mlb_col: "_map_mlb", player_col: "lahman_player_id"}
    if bbref_col:
        rename[bbref_col] = "bbref_id"
    bridge = bridge.rename(columns=rename)
    drop = [c for c in ("lahman_player_id", "bbref_id") if c in out.columns]
    merged = out.drop(columns=drop).merge(
        bridge, left_on="mlb_player_id", right_on="_map_mlb", how="left"
    )
    return merged.drop(columns=["_map_mlb"], errors="ignore")


def load_team_map(path: str | Path = DEFAULT_TEAM_MAP) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"mlb_team_id", "lahman_team_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"mlb team map missing columns: {sorted(missing)}")
    return frame


def attach_team_aliases(teams: pd.DataFrame, team_map: pd.DataFrame) -> pd.DataFrame:
    """Attach MLB + Lahman team ids from the year-aware MLB map via abbreviation."""
    out = teams.copy()
    if "mlb_team_id" not in out.columns:
        out["mlb_team_id"] = pd.NA
    if "lahman_team_id" not in out.columns:
        out["lahman_team_id"] = pd.NA
    if out.empty or team_map.empty:
        return out
    abbr_col = next(
        (c for c in ("mlb_abbr", "abbreviation", "key") if c in team_map.columns),
        None,
    )
    if abbr_col is None:
        return out
    mapped = team_map[[abbr_col, "mlb_team_id", "lahman_team_id"]].drop_duplicates(abbr_col)
    mapped = mapped.rename(columns={abbr_col: "_map_abbr", "mlb_team_id": "_mlb_team_id"})
    if "year_end" in team_map.columns:
        latest = (
            team_map.sort_values("year_end")
            .drop_duplicates(abbr_col, keep="last")
            [[abbr_col, "mlb_team_id", "lahman_team_id"]]
            .rename(columns={abbr_col: "_map_abbr", "mlb_team_id": "_mlb_team_id"})
        )
        mapped = latest
    drop = [c for c in ("mlb_team_id", "lahman_team_id") if c in out.columns]
    merged = out.drop(columns=drop).merge(
        mapped, left_on="sdio_abbr", right_on="_map_abbr", how="left"
    )
    merged = merged.rename(columns={"_mlb_team_id": "mlb_team_id"})
    return merged.drop(columns=["_map_abbr"], errors="ignore")


def build_spine_frames(
    *,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
    player_game: pd.DataFrame,
    player_season: pd.DataFrame,
    as_of_date: str,
    run_id: str | None = None,
    computed_at: str | None = None,
) -> SdioFrames:
    """Mint UUIDs, aliases, and provenance-bearing spine facts."""
    stamp = computed_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run = run_id or default_run_id()
    provenance = {
        "source": SOURCE_NAME,
        "computed_at": stamp,
        "as_of": as_of_date,
        "run_id": run,
        "is_approx": False,
    }

    team_rows = teams.copy()
    if not team_rows.empty:
        team_rows["team_id"] = team_rows["sdio_team_id"].map(lambda v: stable_uuid("team", v))
        team_rows["source_endpoint"] = ENDPOINT_TEAMS
        for key, value in provenance.items():
            team_rows[key] = value

    player_rows = players.copy()
    if not player_rows.empty:
        player_rows["player_id"] = player_rows["sdio_player_id"].map(
            lambda v: stable_uuid("player", v)
        )
        if "sdio_team_id" in player_rows.columns:
            player_rows["team_id"] = player_rows["sdio_team_id"].map(
                lambda v: stable_uuid("team", v) if pd.notna(v) else pd.NA
            )
        player_rows["source_endpoint"] = ENDPOINT_PLAYERS
        for key, value in provenance.items():
            player_rows[key] = value

    game_rows = games.copy()
    if not game_rows.empty:
        game_rows["game_id"] = game_rows["sdio_game_id"].map(lambda v: stable_uuid("game", v))
        if "home_sdio_team_id" in game_rows.columns:
            game_rows["home_team_id"] = game_rows["home_sdio_team_id"].map(
                lambda v: stable_uuid("team", v) if pd.notna(v) else pd.NA
            )
        if "away_sdio_team_id" in game_rows.columns:
            game_rows["away_team_id"] = game_rows["away_sdio_team_id"].map(
                lambda v: stable_uuid("team", v) if pd.notna(v) else pd.NA
            )
        game_rows["source_endpoint"] = ENDPOINT_GAMES_BY_DATE
        for key, value in provenance.items():
            game_rows[key] = value

    aliases = _build_aliases(team_rows, player_rows, game_rows, provenance)

    pgs = player_game.copy()
    if not pgs.empty:
        pgs["player_id"] = pgs["sdio_player_id"].map(lambda v: stable_uuid("player", v))
        pgs["game_id"] = pgs["sdio_game_id"].map(lambda v: stable_uuid("game", v))
        pgs["team_id"] = pgs["sdio_team_id"].map(
            lambda v: stable_uuid("team", v) if pd.notna(v) else pd.NA
        )
        pgs["source_endpoint"] = ENDPOINT_PLAYER_GAME_STATS
        for key, value in provenance.items():
            pgs[key] = value
        pgs = pgs.drop_duplicates(["player_id", "game_id"])

    pss = player_season.copy()
    if not pss.empty:
        pss["player_id"] = pss["sdio_player_id"].map(lambda v: stable_uuid("player", v))
        pss["team_id"] = pss["sdio_team_id"].map(
            lambda v: stable_uuid("team", v) if pd.notna(v) else pd.NA
        )
        pss["source_endpoint"] = ENDPOINT_PLAYER_SEASON_STATS
        for key, value in provenance.items():
            pss[key] = value
        keys = ["player_id", "season", "team_id"]
        pss = pss.dropna(subset=["player_id", "season"]).drop_duplicates(keys)

    # Bootstrap any player/team/game ids that only appear on stats rows.
    player_rows = _ensure_stat_players(player_rows, pgs, pss, provenance)
    team_rows = _ensure_stat_teams(team_rows, pgs, pss, game_rows, provenance)
    game_rows = _ensure_stat_games(game_rows, pgs, provenance)
    aliases = _build_aliases(team_rows, player_rows, game_rows, provenance)

    return SdioFrames(
        as_of_date=as_of_date,
        players=player_rows,
        teams=team_rows,
        games=game_rows,
        aliases=aliases,
        player_game_stat=pgs,
        player_season_stat=pss,
    )


def load_sdio_frames(
    raw_dir: str | Path,
    *,
    as_of_date: str | None = None,
    people: pd.DataFrame | None = None,
    team_map_path: str | Path = DEFAULT_TEAM_MAP,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
    run_id: str | None = None,
) -> SdioFrames:
    """Parse landed SDIO JSON into spine frames. Empty if nothing landed."""
    resolved = resolve_as_of_date(raw_dir, as_of_date=as_of_date, environ=environ)
    if resolved is None:
        return SdioFrames()
    try:
        team_map = load_team_map(team_map_path)
    except (OSError, ValueError) as exc:
        log.warning("Team map unavailable (%s); continuing without MLB/Lahman team aliases", exc)
        team_map = pd.DataFrame()

    teams = _read_parsed(parse_teams, ENDPOINT_TEAMS, "teams.json", raw_dir, resolved, backend)
    if not teams.empty:
        teams = attach_team_aliases(teams, team_map)

    players = _read_parsed(
        parse_players, ENDPOINT_PLAYERS, "players.json", raw_dir, resolved, backend
    )
    if not players.empty:
        players = attach_lahman_aliases(players, people)

    game_parts: list[pd.DataFrame] = []
    pgs_parts: list[pd.DataFrame] = []
    pss_parts: list[pd.DataFrame] = []

    games_by_date = _read_parsed(
        parse_games,
        ENDPOINT_GAMES_BY_DATE,
        f"games_by_date_{resolved}.json",
        raw_dir,
        resolved,
        backend,
    )
    if not games_by_date.empty:
        game_parts.append(games_by_date)
    pgs = _read_parsed(
        parse_player_game_stats,
        ENDPOINT_PLAYER_GAME_STATS,
        f"player_game_stats_{resolved}.json",
        raw_dir,
        resolved,
        backend,
    )
    if not pgs.empty:
        pgs_parts.append(pgs)

    years = _discover_seasons(raw_dir, resolved, backend=backend)
    for year in years:
        season_games = _read_parsed(
            parse_games,
            ENDPOINT_GAMES,
            f"games_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        if not season_games.empty:
            game_parts.append(season_games)
        season_stats = _read_parsed(
            parse_player_season_stats,
            ENDPOINT_PLAYER_SEASON_STATS,
            f"player_season_stats_{year}.json",
            raw_dir,
            resolved,
            backend,
        )
        if not season_stats.empty:
            pss_parts.append(season_stats)

    games = (
        pd.concat(game_parts, ignore_index=True).drop_duplicates("sdio_game_id")
        if game_parts
        else pd.DataFrame()
    )
    player_game = pd.concat(pgs_parts, ignore_index=True) if pgs_parts else pd.DataFrame()
    player_season = pd.concat(pss_parts, ignore_index=True) if pss_parts else pd.DataFrame()

    return build_spine_frames(
        teams=teams,
        players=players,
        games=games,
        player_game=player_game,
        player_season=player_season,
        as_of_date=resolved,
        run_id=run_id,
        computed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


def default_season_window(as_of_date: str) -> list[int]:
    """Inclusive ``[Y-2, Y]`` from the ``as_of_date`` calendar year.

    ``Y`` is the MLB championship season year implied by the extract cut
    date, not a hardcoded 2024–2026 forever. A run dated 2026-08-23
    therefore pulls 2024, 2025, and 2026.
    """
    text = str(as_of_date).strip()
    if len(text) < 4 or not text[:4].isdigit():
        raise ValueError(f"as_of_date must start with a 4-digit year, got {as_of_date!r}")
    year = int(text[:4])
    return list(range(year - 2, year + 1))


def mark_extract_season_coverage(report: ExtractReport) -> ExtractReport:
    """Flag when the active season did not land. Soft-fail is not “current.”"""
    window = list(report.seasons) or default_season_window(report.as_of_date)
    active = max(window)
    report.active_season = active
    if report.skipped_reason == "missing_api_key" or not report.endpoints:
        report.current_season_missing = True
        return report
    landed_season = any(
        item.ok
        and item.endpoint == ENDPOINT_PLAYER_SEASON_STATS
        and item.season == active
        for item in report.endpoints
    )
    landed_games = int(str(report.as_of_date)[:4]) == active and any(
        item.ok and item.endpoint == ENDPOINT_PLAYER_GAME_STATS
        for item in report.endpoints
    )
    report.current_season_missing = not (landed_season or landed_games)
    return report


def seasons_from_settings(
    settings: Mapping[str, Any] | None,
    as_of_date: str,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[int]:
    env = os.environ if environ is None else environ
    raw_env = (env.get("SPORTSDATAIO_SEASONS") or "").strip()
    if raw_env:
        return sorted({int(part) for part in raw_env.split(",") if part.strip()})
    configured = (settings or {}).get("sportsdataio") or {}
    years = configured.get("seasons") or []
    if years:
        return sorted({int(year) for year in years})
    return default_season_window(as_of_date)


def client_from_settings(
    settings: Mapping[str, Any] | None = None,
    *,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> SportsDataIOClient:
    configured = (settings or {}).get("sportsdataio") or {}
    return SportsDataIOClient(
        api_key=api_key,
        base_url=str(configured.get("base_url") or SDIO_API_BASE),
        min_interval=float(configured.get("min_request_interval") or DEFAULT_MIN_INTERVAL_S),
        max_retries=int(configured.get("max_retries") or DEFAULT_MAX_RETRIES),
        environ=environ,
    )


def open_optional_backend(
    uri: str | None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ArtifactBackend | None:
    if not uri:
        return None
    return open_backend(uri, environ=environ)


def _write_report(
    report: ExtractReport,
    raw_dir: str | Path,
    as_of_date: str,
    backend: ArtifactBackend | None,
) -> None:
    write_raw_payload(
        report.to_dict(),
        endpoint=ENDPOINT_EXTRACT_REPORT,
        as_of_date=as_of_date,
        filename="extract_report.json",
        raw_dir=raw_dir,
        backend=backend,
    )


def _pull_one(
    fetch: Callable[[], Any],
    *,
    endpoint: str,
    filename: str,
    raw_dir: str | Path,
    as_of_date: str,
    backend: ArtifactBackend | None,
    season: int | None = None,
    date: str | None = None,
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
            as_of_date=date or as_of_date,
        )
    except Exception as exc:  # soft-fail a single endpoint
        log.warning("SportsDataIO %s failed softly: %s", endpoint, exc)
        return EndpointResult(
            endpoint=endpoint,
            ok=False,
            error=str(exc),
            season=season,
            as_of_date=date or as_of_date,
        )


def _build_aliases(
    teams: pd.DataFrame,
    players: pd.DataFrame,
    games: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        *,
        entity_type: str,
        internal_id: object,
        system: str,
        external_id: object,
        is_primary: bool,
        source_endpoint: str,
    ) -> None:
        if pd.isna(internal_id) or pd.isna(external_id) or external_id in {"", None}:
            return
        if system not in ALIAS_SYSTEMS or entity_type not in ENTITY_TYPES:
            return
        ext = str(int(external_id)) if _is_intish(external_id) else str(external_id).strip()
        if not ext:
            return
        rows.append(
            {
                "alias_id": stable_uuid("alias", f"{system}:{entity_type}:{ext}"),
                "entity_type": entity_type,
                "internal_id": str(internal_id),
                "system": system,
                "external_id": ext,
                "is_primary": bool(is_primary),
                "source_endpoint": source_endpoint,
                **dict(provenance),
            }
        )

    if not teams.empty:
        for row in teams.to_dict(orient="records"):
            add(
                entity_type="team",
                internal_id=row.get("team_id"),
                system="sportsdataio",
                external_id=row.get("sdio_team_id"),
                is_primary=True,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_TEAMS,
            )
            add(
                entity_type="team",
                internal_id=row.get("team_id"),
                system="mlb",
                external_id=row.get("mlb_team_id"),
                is_primary=False,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_TEAMS,
            )
            add(
                entity_type="team",
                internal_id=row.get("team_id"),
                system="lahman",
                external_id=row.get("lahman_team_id"),
                is_primary=False,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_TEAMS,
            )
    if not players.empty:
        for row in players.to_dict(orient="records"):
            add(
                entity_type="player",
                internal_id=row.get("player_id"),
                system="sportsdataio",
                external_id=row.get("sdio_player_id"),
                is_primary=True,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_PLAYERS,
            )
            add(
                entity_type="player",
                internal_id=row.get("player_id"),
                system="mlb",
                external_id=row.get("mlb_player_id"),
                is_primary=False,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_PLAYERS,
            )
            add(
                entity_type="player",
                internal_id=row.get("player_id"),
                system="bbref",
                external_id=row.get("bbref_id"),
                is_primary=False,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_PLAYERS,
            )
            add(
                entity_type="player",
                internal_id=row.get("player_id"),
                system="lahman",
                external_id=row.get("lahman_player_id"),
                is_primary=False,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_PLAYERS,
            )
    if not games.empty:
        for row in games.to_dict(orient="records"):
            add(
                entity_type="game",
                internal_id=row.get("game_id"),
                system="sportsdataio",
                external_id=row.get("sdio_game_id"),
                is_primary=True,
                source_endpoint=row.get("source_endpoint") or ENDPOINT_GAMES_BY_DATE,
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.drop_duplicates(["system", "entity_type", "external_id"]).reset_index(drop=True)


def _ensure_stat_players(
    players: pd.DataFrame,
    pgs: pd.DataFrame,
    pss: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    known = set(players["sdio_player_id"]) if not players.empty and "sdio_player_id" in players.columns else set()
    extras: list[dict[str, Any]] = []
    for frame in (pgs, pss):
        if frame.empty or "sdio_player_id" not in frame.columns:
            continue
        for row in frame.to_dict(orient="records"):
            sid = row.get("sdio_player_id")
            if sid in known or pd.isna(sid):
                continue
            known.add(sid)
            extras.append(
                {
                    "sdio_player_id": sid,
                    "sdio_team_id": row.get("sdio_team_id"),
                    "display_name": row.get("display_name"),
                    "position": row.get("position"),
                    "player_id": stable_uuid("player", sid),
                    "team_id": (
                        stable_uuid("team", row["sdio_team_id"])
                        if pd.notna(row.get("sdio_team_id"))
                        else pd.NA
                    ),
                    "source_endpoint": row.get("source_endpoint") or ENDPOINT_PLAYER_GAME_STATS,
                    **dict(provenance),
                }
            )
    if not extras:
        return players
    extra_df = pd.DataFrame(extras)
    return pd.concat([players, extra_df], ignore_index=True) if not players.empty else extra_df


def _ensure_stat_teams(
    teams: pd.DataFrame,
    pgs: pd.DataFrame,
    pss: pd.DataFrame,
    games: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    known = set(teams["sdio_team_id"]) if not teams.empty and "sdio_team_id" in teams.columns else set()
    extras: list[dict[str, Any]] = []

    def consider(sid: object, abbr: object | None = None, endpoint: str = ENDPOINT_TEAMS) -> None:
        if sid in known or pd.isna(sid):
            return
        known.add(sid)
        extras.append(
            {
                "sdio_team_id": sid,
                "sdio_abbr": abbr,
                "team_id": stable_uuid("team", sid),
                "source_endpoint": endpoint,
                **dict(provenance),
            }
        )

    for frame, col in (
        (pgs, "sdio_team_id"),
        (pss, "sdio_team_id"),
        (games, "home_sdio_team_id"),
        (games, "away_sdio_team_id"),
    ):
        if frame.empty or col not in frame.columns:
            continue
        for row in frame.to_dict(orient="records"):
            consider(row.get(col), endpoint=row.get("source_endpoint") or ENDPOINT_TEAMS)
    if not extras:
        return teams
    extra_df = pd.DataFrame(extras)
    return pd.concat([teams, extra_df], ignore_index=True) if not teams.empty else extra_df


def _ensure_stat_games(
    games: pd.DataFrame,
    pgs: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> pd.DataFrame:
    known = set(games["sdio_game_id"]) if not games.empty and "sdio_game_id" in games.columns else set()
    extras: list[dict[str, Any]] = []
    if pgs.empty or "sdio_game_id" not in pgs.columns:
        return games
    for row in pgs.to_dict(orient="records"):
        sid = row.get("sdio_game_id")
        if sid in known or pd.isna(sid):
            continue
        known.add(sid)
        extras.append(
            {
                "sdio_game_id": sid,
                "game_date": row.get("game_date"),
                "season": row.get("season"),
                "game_id": stable_uuid("game", sid),
                "source_endpoint": ENDPOINT_PLAYER_GAME_STATS,
                **dict(provenance),
            }
        )
    if not extras:
        return games
    extra_df = pd.DataFrame(extras)
    return pd.concat([games, extra_df], ignore_index=True) if not games.empty else extra_df


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
    if isinstance(report, Mapping):
        for year in report.get("seasons") or []:
            years.add(int(year))
        for item in report.get("endpoints") or []:
            if item.get("season"):
                years.add(int(item["season"]))
    return sorted(years) or [int(as_of_date[:4])]


def _read_parsed(
    parser: Callable[[Any], pd.DataFrame],
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
    if payload is None:
        return pd.DataFrame()
    try:
        return parser(payload)
    except Exception as exc:
        log.warning("Failed to parse %s/%s: %s", endpoint, filename, exc)
        return pd.DataFrame()


def _as_records(payload: Any) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        for key in ("value", "data", "teams", "players", "games", "stats"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, Mapping)]
        if any(key in payload for key in ("TeamID", "PlayerID", "GameID")):
            return [payload]
    return []


def _year_from_filename(name: str) -> int | None:
    for part in Path(name).stem.split("_"):
        if part.isdigit() and len(part) == 4:
            return int(part)
    return None


def _endpoint_token(value: str) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid SportsDataIO endpoint token: {value!r}")
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


def _is_intish(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    text = str(value).strip()
    return text.lstrip("-").isdigit()


def _drop_null_id(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    return df[df[column].notna()].reset_index(drop=True)


def _backoff(attempt: int) -> None:
    time.sleep(min(8.0, 0.5 * (2 ** attempt)))
