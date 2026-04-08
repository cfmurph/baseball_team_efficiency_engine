"""
Sportradar MLB API v8 client.

Authentication:
    Set the SPORTRADAR_API_KEY environment variable, or pass api_key
    explicitly to SportradarClient().

Access levels:
    trial       – free trial key (1 req/sec, limited history)
    production  – paid subscription

Usage::

    from src.baseball_analytics.sportradar import SportradarClient

    client = SportradarClient()          # reads SPORTRADAR_API_KEY from env
    hierarchy = client.league_hierarchy()
    stats = client.seasonal_stats(team_id="...", year=2024)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

# Load .env if python-dotenv is installed. Gracefully skipped otherwise.
# override=False means shell exports and Cursor Cloud Secrets always win.
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except ImportError:
    pass

log = logging.getLogger(__name__)

_BASE = "https://api.sportradar.com/mlb"
_DEFAULT_ACCESS = "trial"
_DEFAULT_LANG = "en"
_DEFAULT_FMT = "json"

# Sportradar trial keys are rate-limited to 1 req/sec.
# Production keys allow higher throughput; the client respects _min_interval.
_TRIAL_MIN_INTERVAL = 1.1   # seconds between requests on trial key
_PROD_MIN_INTERVAL = 0.1


class SportradarError(Exception):
    """Raised for non-2xx API responses."""
    def __init__(self, status_code: int, url: str, body: str = "") -> None:
        self.status_code = status_code
        self.url = url
        super().__init__(f"HTTP {status_code} from {url}: {body[:200]}")


class SportradarClient:
    """
    Thin wrapper around the Sportradar MLB v8 REST API.

    Parameters
    ----------
    api_key : str, optional
        API key.  Falls back to the SPORTRADAR_API_KEY environment variable.
    access_level : str
        'trial' or 'production' (or 'tracking' / 'trial-tracking' for Statcast).
    cache_dir : Path or str, optional
        If set, raw JSON responses are cached to disk at this path.
        Subsequent identical calls return the cached file instead of
        hitting the network.  Useful for development and testing.
    max_retries : int
        Number of retry attempts on transient 429 / 5xx errors.
    """

    def __init__(
        self,
        api_key: str | None = None,
        access_level: str = _DEFAULT_ACCESS,
        cache_dir: Path | str | None = None,
        max_retries: int = 3,
    ) -> None:
        key = api_key or os.environ.get("SPORTRADAR_API_KEY")
        if not key:
            raise EnvironmentError(
                "Sportradar API key not found. "
                "Set the SPORTRADAR_API_KEY environment variable or pass api_key=."
            )
        self._key = key
        self._access = access_level
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"x-api-key": self._key})
        self._last_request_time: float = 0.0
        self._min_interval = (
            _TRIAL_MIN_INTERVAL if access_level.startswith("trial") else _PROD_MIN_INTERVAL
        )

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, path: str) -> str:
        return path.replace("/", "_").strip("_") + ".json"

    def _get(self, path: str) -> dict[str, Any]:
        """Execute a GET request with rate limiting, retrying, and optional caching."""
        if self._cache_dir:
            cached = self._cache_dir / self._cache_key(path)
            if cached.exists():
                import json
                log.debug("Cache hit: %s", cached)
                return json.loads(cached.read_text())

        url = f"{_BASE}/{self._access}/v8/{_DEFAULT_LANG}/{path}.{_DEFAULT_FMT}"

        for attempt in range(self._max_retries + 1):
            # Rate limiting
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)

            log.debug("GET %s (attempt %d)", url, attempt + 1)
            resp = self._session.get(url, timeout=30)
            self._last_request_time = time.monotonic()

            if resp.status_code == 200:
                data = resp.json()
                if self._cache_dir:
                    import json
                    (self._cache_dir / self._cache_key(path)).write_text(
                        json.dumps(data, indent=2)
                    )
                return data

            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** attempt
                log.warning("HTTP %d — retrying in %ds", resp.status_code, wait)
                time.sleep(wait)
                continue

            raise SportradarError(resp.status_code, url, resp.text)

        raise SportradarError(resp.status_code, url, resp.text)  # type: ignore[possibly-undefined]

    # ------------------------------------------------------------------
    # Endpoint methods
    # ------------------------------------------------------------------

    def league_hierarchy(self) -> dict[str, Any]:
        """
        Returns all teams with league/division info and venue data.
        Endpoint: /league/hierarchy.json
        """
        return self._get("league/hierarchy")

    def seasons(self) -> dict[str, Any]:
        """
        Returns list of all available seasons.
        Endpoint: /league/seasons.json
        """
        return self._get("league/seasons")

    def seasonal_stats(self, team_id: str, year: int, season_type: str = "REG") -> dict[str, Any]:
        """
        Season-to-date hitting, pitching, and fielding stats for all players
        on the given team's roster.

        Parameters
        ----------
        team_id : str
            Sportradar team GUID.
        year : int
            Season year (e.g. 2024).
        season_type : str
            'REG' (regular season), 'PRE' (spring training), 'PST' (postseason).

        Endpoint: /seasons/{year}/{season_type}/teams/{team_id}/statistics.json
        """
        return self._get(f"seasons/{year}/{season_type}/teams/{team_id}/statistics")

    def transactions(self, year: int, month: int, day: int) -> dict[str, Any]:
        """
        All transactions for a given date.
        Endpoint: /league/{year}/{month:02d}/{day:02d}/transactions.json
        """
        return self._get(f"league/{year}/{month:02d}/{day:02d}/transactions")

    def injuries(self) -> dict[str, Any]:
        """
        Current injury report for all MLB players.
        Endpoint: /league/injuries.json
        """
        return self._get("league/injuries")

    def standings(self, year: int, season_type: str = "REG") -> dict[str, Any]:
        """
        Division standings for a given season.
        Endpoint: /seasons/{year}/{season_type}/standings.json
        """
        return self._get(f"seasons/{year}/{season_type}/standings")

    def league_schedule(self, year: int, season_type: str = "REG") -> dict[str, Any]:
        """
        Full schedule for a given season.
        Endpoint: /games/{year}/{season_type}/schedule.json
        """
        return self._get(f"games/{year}/{season_type}/schedule")

    def team_profile(self, team_id: str) -> dict[str, Any]:
        """
        Team roster and player info.
        Endpoint: /teams/{team_id}/profile.json
        """
        return self._get(f"teams/{team_id}/profile")
