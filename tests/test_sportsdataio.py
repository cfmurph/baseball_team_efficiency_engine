"""Extract, alias, spine, and soft-fail tests for SportsDataIO Phase 0 (#128)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest

from pipeline.extract import pull_sportsdataio as pull_mod
from pipeline.transform.build_warehouse import insert_sdio_spine_tables
from src.baseball_analytics.schema import WAREHOUSE_DDL

from src.baseball_analytics.sportsdataio import (
    API_KEY_ENV,
    ENDPOINT_TEAMS,
    RAW_REMOTE_PREFIX,
    MissingApiKeyError,
    SportsDataIOClient,
    SportsDataIOError,
    attach_lahman_aliases,
    default_season_window,
    discover_as_of_dates,
    extract_had_in_season,
    load_sdio_frames,
    load_team_map,
    local_raw_path,
    parse_games,
    parse_player_game_stats,
    parse_player_season_stats,
    parse_players,
    parse_teams,
    pull_phase0_feeds,
    raw_object_key,
    read_raw_payload,
    resolve_api_key,
    resolve_as_of_date,
    sdio_date_token,
    seasons_from_settings,
    stable_uuid,
    write_raw_payload,
)
from src.baseball_analytics.storage import FileBackend

FIXTURES = Path(__file__).parent / "fixtures" / "sportsdataio"
TEAM_MAP = Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "mlb_team_map.csv"
AS_OF = "2026-08-23"


def _payload(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _land_fixtures(raw_dir: Path, as_of: str = AS_OF, backend: FileBackend | None = None) -> None:
    mapping = {
        ("teams", "teams.json"): "teams.json",
        ("players", "players.json"): "players.json",
        ("games_by_date", f"games_by_date_{as_of}.json"): "games_by_date.json",
        ("player_game_stats", f"player_game_stats_{as_of}.json"): "player_game_stats.json",
        ("player_season_stats", "player_season_stats_2024.json"): "player_season_stats.json",
    }
    for (endpoint, filename), fixture in mapping.items():
        write_raw_payload(
            _payload(fixture),
            endpoint=endpoint,
            as_of_date=as_of,
            filename=filename,
            raw_dir=raw_dir,
            backend=backend,
        )
    write_raw_payload(
        {"as_of_date": as_of, "seasons": [2024], "ok": True, "endpoints": [], "schema_version": "0.1"},
        endpoint="extract_report",
        as_of_date=as_of,
        filename="extract_report.json",
        raw_dir=raw_dir,
        backend=backend,
    )


@pytest.mark.unit
def test_raw_object_key_matches_locked_layout() -> None:
    key = raw_object_key("player_game_stats", AS_OF, "player_game_stats_2026-08-23.json")
    assert key == f"{RAW_REMOTE_PREFIX}/player_game_stats/{AS_OF}/player_game_stats_2026-08-23.json"
    local = local_raw_path("data/raw", "teams", AS_OF, "teams.json")
    assert local.as_posix().endswith(f"data/raw/sportsdataio/teams/{AS_OF}/teams.json")


@pytest.mark.unit
def test_default_season_window_is_y_minus_2_through_y() -> None:
    assert default_season_window("2026-08-23") == [2024, 2025, 2026]
    assert default_season_window("2027-04-01") == [2025, 2026, 2027]


@pytest.mark.unit
def test_seasons_from_settings_defaults_to_window_and_honors_overrides() -> None:
    assert seasons_from_settings({}, "2026-08-23", environ={}) == [2024, 2025, 2026]
    assert seasons_from_settings(
        {"sportsdataio": {"seasons": []}}, "2026-12-31", environ={}
    ) == [2024, 2025, 2026]
    assert seasons_from_settings(
        {"sportsdataio": {"seasons": [2023, 2024]}},
        "2026-08-23",
        environ={},
    ) == [2023, 2024]
    assert seasons_from_settings(
        {"sportsdataio": {"seasons": [2023]}},
        "2026-08-23",
        environ={"SPORTSDATAIO_SEASONS": "2024,2026"},
    ) == [2024, 2026]


@pytest.mark.unit
def test_sdio_date_token_uses_month_abbrev() -> None:
    assert sdio_date_token("2024-07-31") == "2024-JUL-31"
    assert sdio_date_token("2017-09-01") == "2017-SEP-01"


@pytest.mark.unit
def test_stable_uuid_is_deterministic() -> None:
    first = stable_uuid("player", 10001967)
    second = stable_uuid("player", 10001967)
    assert first == second
    assert first != stable_uuid("team", 10001967)


@pytest.mark.unit
def test_resolve_api_key_reads_env_only() -> None:
    assert resolve_api_key(environ={}) is None
    assert resolve_api_key(environ={API_KEY_ENV: "  secret-key  "}) == "secret-key"
    assert resolve_api_key("cli-key", environ={API_KEY_ENV: "env-key"}) == "cli-key"


@pytest.mark.unit
def test_parse_teams_and_players() -> None:
    teams = parse_teams(_payload("teams.json"))
    assert set(teams["sdio_team_id"]) == {31, 20}
    yankees = teams.set_index("sdio_team_id").loc[31]
    assert yankees["sdio_abbr"] == "NYY"
    players = parse_players(_payload("players.json"))
    judge = players.set_index("sdio_player_id").loc[10001967]
    assert judge["display_name"] == "Aaron Judge"
    assert int(judge["mlb_player_id"]) == 592450


@pytest.mark.unit
def test_parse_games_and_player_game_stats() -> None:
    games = parse_games(_payload("games_by_date.json"))
    assert int(games.iloc[0]["sdio_game_id"]) == 74546
    assert int(games.iloc[0]["home_score"]) == 7
    stats = parse_player_game_stats(_payload("player_game_stats.json"))
    judge = stats.set_index("sdio_player_id").loc[10001967]
    assert judge["hr"] == 1
    assert judge["pa"] == 5
    assert int(judge["sdio_game_id"]) == 74546
    season = parse_player_season_stats(_payload("player_season_stats.json"))
    assert int(season.iloc[0]["games"]) == 158


@pytest.mark.unit
def test_attach_mlb_and_bbref_aliases_via_people() -> None:
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01", "ohtansh01"],
            "mlbID": [592450, 660271],
            "bbrefID": ["judgeaa01", "ohtani-shohei"],
        }
    )
    players = parse_players(_payload("players.json"))
    joined = attach_lahman_aliases(players, people)
    judge = joined.set_index("sdio_player_id").loc[10001967]
    assert judge["lahman_player_id"] == "judgeaa01"
    assert judge["bbref_id"] == "judgeaa01"


@pytest.mark.integration
def test_extract_writes_local_and_file_uri(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    lake = tmp_path / "lake"
    backend = FileBackend(lake)

    def fetcher(path: str, params: dict):
        if path.endswith("/Teams"):
            return _payload("teams.json")
        if path.endswith("/Players"):
            return _payload("players.json")
        if "GamesByDate" in path:
            return _payload("games_by_date.json")
        if "PlayerGameStatsByDate" in path:
            return _payload("player_game_stats.json")
        if "PlayerSeasonStats" in path:
            return _payload("player_season_stats.json")
        raise SportsDataIOError(f"unexpected path {path}")

    client = SportsDataIOClient(api_key="test-key", fetcher=fetcher, min_interval=0)
    report = pull_phase0_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
        backend=backend,
    )
    assert report.ok
    local_teams = local_raw_path(raw_dir, "teams", AS_OF, "teams.json")
    assert local_teams.is_file()
    remote_teams = lake / raw_object_key("teams", AS_OF, "teams.json")
    assert remote_teams.is_file()
    assert json.loads(local_teams.read_text()) == json.loads(remote_teams.read_text())
    assert (lake / raw_object_key("extract_report", AS_OF, "extract_report.json")).is_file()
    assert discover_as_of_dates(raw_dir) == [AS_OF]
    assert raw_object_key("teams", AS_OF, "teams.json").startswith("raw/sportsdataio/")


@pytest.mark.integration
def test_extract_same_date_overwrite_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_payload(
        [{"TeamID": 1}],
        endpoint=ENDPOINT_TEAMS,
        as_of_date=AS_OF,
        filename="teams.json",
        raw_dir=raw_dir,
    )
    write_raw_payload(
        _payload("teams.json"),
        endpoint=ENDPOINT_TEAMS,
        as_of_date=AS_OF,
        filename="teams.json",
        raw_dir=raw_dir,
    )
    landed = read_raw_payload(
        endpoint=ENDPOINT_TEAMS,
        as_of_date=AS_OF,
        filename="teams.json",
        raw_dir=raw_dir,
    )
    assert landed is not None
    assert len(landed) == 2


@pytest.mark.unit
def test_extract_had_in_season_from_season_rows_or_game_payload() -> None:
    in_season = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": None,
        "current_season_missing": False,
        "endpoints": [
            {"endpoint": "player_season_stats", "ok": True, "season": 2026},
        ],
    }
    games_only = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": None,
        "endpoints": [
            {"endpoint": "player_game_stats", "ok": True, "season": None},
        ],
    }
    missing_key = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": "missing_api_key",
        "current_season_missing": True,
        "endpoints": [],
    }
    empty = {
        "as_of_date": AS_OF,
        "seasons": [2024, 2025, 2026],
        "skipped_reason": None,
        "current_season_missing": True,
        "endpoints": [
            {"endpoint": "player_season_stats", "ok": True, "season": 2025},
        ],
    }
    assert extract_had_in_season(in_season, active_season=2026) is True
    assert extract_had_in_season(games_only, active_season=2026) is True
    assert extract_had_in_season(missing_key, active_season=2026) is False
    assert extract_had_in_season(empty, active_season=2026) is False
    assert extract_had_in_season(None, active_season=2026) is False


@pytest.mark.integration
def test_extract_soft_fails_without_api_key(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    client = SportsDataIOClient(api_key=None, environ={}, min_interval=0)
    report = pull_phase0_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
    )
    assert report.ok is False
    assert report.soft_fail is True
    assert report.skipped_reason == "missing_api_key"
    assert report.endpoints == []
    assert report.active_season == 2024
    assert report.current_season_missing is True
    landed = read_raw_payload(
        endpoint="extract_report",
        as_of_date=AS_OF,
        filename="extract_report.json",
        raw_dir=raw_dir,
    )
    assert landed["ok"] is False
    assert landed["soft_fail"] is True
    assert landed["current_season_missing"] is True
    assert landed["active_season"] == 2024


@pytest.mark.integration
def test_extract_soft_fails_on_api_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    def fetcher(path: str, params: dict):
        if path.endswith("/Teams"):
            return _payload("teams.json")
        raise SportsDataIOError("HTTP 503 from SportsDataIO", status_code=503, url=path)

    client = SportsDataIOClient(api_key="test-key", fetcher=fetcher, min_interval=0)
    report = pull_phase0_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
    )
    assert report.ok is False
    assert any(item.endpoint == "teams" and item.ok for item in report.endpoints)
    assert any(item.endpoint == "player_game_stats" and not item.ok for item in report.endpoints)
    assert local_raw_path(raw_dir, "teams", AS_OF, "teams.json").is_file()


@pytest.mark.integration
def test_cli_soft_fail_without_key_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "raw_dir: raw\nartifacts_uri: ''\nartifacts_dir: artifacts\nsportsdataio: {}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    result = CliRunner().invoke(
        pull_mod.app,
        ["--config-path", str(settings_path), "--as-of-date", AS_OF],
    )
    assert result.exit_code == 0
    report = json.loads(
        (tmp_path / "raw" / "sportsdataio" / "extract_report" / AS_OF / "extract_report.json").read_text()
    )
    assert report["ok"] is False
    assert report["soft_fail"] is True
    assert report["current_season_missing"] is True
    assert report["active_season"] == 2026
    assert report["seasons"] == [2024, 2025, 2026]
    assert "SPORTSDATAIO_API_KEY" in (report.get("error") or "")


@pytest.mark.integration
def test_cli_soft_fail_exits_zero_on_boom(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "raw_dir: raw\nartifacts_uri: ''\nartifacts_dir: artifacts\nsportsdataio: {}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("sportsdataio down")

    monkeypatch.setattr(pull_mod, "pull_phase0_feeds", boom)
    result = CliRunner().invoke(
        pull_mod.app,
        ["--config-path", str(settings_path), "--as-of-date", AS_OF],
    )
    assert result.exit_code == 0
    report = json.loads(
        (tmp_path / "raw" / "sportsdataio" / "extract_report" / AS_OF / "extract_report.json").read_text()
    )
    assert report["ok"] is False
    assert report["soft_fail"] is True


@pytest.mark.integration
def test_warehouse_builds_without_sdio(tmp_path: Path) -> None:
    frames = load_sdio_frames(tmp_path / "missing-raw", as_of_date=AS_OF)
    assert frames.empty
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    loaded = insert_sdio_spine_tables(con, frames)
    assert loaded == {}
    count = con.execute("SELECT COUNT(*) FROM player_game_stat").fetchone()[0]
    assert count == 0
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "external_id_alias" in tables
    assert "player_game_stat" in tables
    assert not any(name.startswith("fantasy_") and name.endswith("_stat") for name in tables)
    assert not any(name.startswith("scout_") and name.endswith("_stat") for name in tables)


@pytest.mark.integration
def test_warehouse_loads_spine_and_aliases(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _land_fixtures(raw_dir)
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01", "ohtansh01"],
            "nameFirst": ["Aaron", "Shohei"],
            "nameLast": ["Judge", "Ohtani"],
            "mlbID": [592450, 660271],
            "bbrefID": ["judgeaa01", "ohtani-shohei"],
        }
    )
    frames = load_sdio_frames(
        raw_dir,
        as_of_date=AS_OF,
        people=people,
        team_map_path=TEAM_MAP,
        run_id="test-run",
    )
    assert not frames.empty
    judge = frames.players.set_index("sdio_player_id").loc[10001967]
    assert judge["player_id"] == stable_uuid("player", 10001967)
    assert judge["lahman_player_id"] == "judgeaa01"
    assert judge["bbref_id"] == "judgeaa01"
    yankees = frames.teams.set_index("sdio_team_id").loc[31]
    assert yankees["lahman_team_id"] == "NYA"
    assert int(yankees["mlb_team_id"]) == 147
    aliases = frames.aliases
    systems = set(aliases["system"])
    assert systems >= {"sportsdataio", "mlb", "bbref", "lahman"}
    judge_sdio = aliases[
        (aliases["system"] == "sportsdataio")
        & (aliases["entity_type"] == "player")
        & (aliases["external_id"] == "10001967")
    ].iloc[0]
    assert bool(judge_sdio["is_primary"])
    assert judge_sdio["internal_id"] == judge["player_id"]
    judge_bbref = aliases[
        (aliases["system"] == "bbref") & (aliases["external_id"] == "judgeaa01")
    ].iloc[0]
    assert judge_bbref["internal_id"] == judge["player_id"]
    pgs = frames.player_game_stat.set_index("player_id").loc[judge["player_id"]]
    assert pgs["hr"] == 1
    assert pgs["game_id"] == stable_uuid("game", 74546)
    for col in ("source", "source_endpoint", "computed_at", "as_of", "run_id", "is_approx"):
        assert col in frames.player_game_stat.columns
        assert pgs[col] is not None
    assert pgs["source"] == "sportsdataio"
    assert bool(pgs["is_approx"]) is False
    assert not frames.player_season_stat.empty

    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.execute(
        "INSERT INTO fact_player_season (player_id, season_key, team_id, player_war, war_source) "
        "VALUES ('judgeaa01', 2024, 'NYA', 10.8, 'real')"
    )
    loaded = insert_sdio_spine_tables(con, frames)
    assert loaded["player"] >= 1
    assert loaded["external_id_alias"] >= 1
    assert loaded["player_game_stat"] >= 1
    war = con.execute(
        "SELECT player_war, war_source FROM fact_player_season WHERE player_id = 'judgeaa01'"
    ).fetchone()
    assert war == (10.8, "real")
    grain = con.execute(
        "SELECT COUNT(*) FROM (SELECT player_id, game_id FROM player_game_stat GROUP BY 1, 2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert grain == 0
    tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    assert "dim_team" in tables
    assert "fact_mlb_game" in tables
    assert "fantasy_player_stat" not in tables
    assert "scout_player_stat" not in tables


@pytest.mark.integration
def test_load_sdio_frames_reads_file_uri_when_local_missing(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    backend = FileBackend(lake)
    _land_fixtures(tmp_path / "seed", backend=backend)
    empty_local = tmp_path / "empty-raw"
    empty_local.mkdir()
    frames = load_sdio_frames(
        empty_local,
        as_of_date=AS_OF,
        team_map_path=TEAM_MAP,
        backend=backend,
    )
    assert not frames.teams.empty
    assert int(frames.teams.iloc[0]["sdio_team_id"]) in {31, 20}


@pytest.mark.unit
def test_sdio_probe_workflow_is_dispatch_only() -> None:
    text = Path(".github/workflows/sdio-probe.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "pull_request:" not in text
    assert "SPORTSDATAIO_API_KEY: ${{ secrets.SPORTSDATAIO_API_KEY }}" in text
    assert "Ocp-Apim-Subscription-Key" in text
    assert 'print("SPORTSDATAIO_API_KEY missing")' in text
    assert "sys.exit(1)" in text
    assert "soft-fail" not in text.split("if not key:", 1)[-1].split("url =", 1)[0]
    assert "/v3/mlb/scores/json/Teams" in text
    assert "HTTPError" in text
    assert "http_status={int(exc.code)}" in text
    assert "print(key" not in text
    assert "?key=" not in text
    nightly = Path(".github/workflows/nightly-refresh.yml").read_text(encoding="utf-8")
    assert "SPORTSDATAIO_API_KEY: ${{ secrets.SPORTSDATAIO_API_KEY }}" in nightly
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "secrets.SPORTSDATAIO_API_KEY" not in ci
    assert not Path(".github/workflows/ci-smoke.yml").exists()


@pytest.mark.unit
def test_warehouse_ddl_has_no_forked_stat_tables() -> None:
    assert "CREATE OR REPLACE TABLE fantasy_" not in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE scout_" not in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE external_id_alias" in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE player_game_stat" in WAREHOUSE_DDL
    assert "PRIMARY KEY (player_id, game_id)" in WAREHOUSE_DDL
    assert "UNIQUE (system, entity_type, external_id)" in WAREHOUSE_DDL


def test_parse_unwraps_payloads_and_drops_rows_missing_ids() -> None:
    wrapped = parse_teams({"data": [{"TeamID": 31, "Key": "NYY"}, {"Key": "NOID"}]})
    assert list(wrapped["sdio_team_id"]) == [31]
    single = parse_teams({"TeamID": 20, "Key": "LAD"})
    assert list(single["sdio_team_id"]) == [20]
    assert parse_teams([]).empty
    assert parse_teams(None).empty
    assert parse_players([{"FirstName": "NoId"}]).empty
    stats = parse_player_game_stats(
        [
            {"Name": "Missing both"},
            {"PlayerID": 1, "Name": "Missing game"},
            {"GameID": 2, "Name": "Missing player"},
            {"PlayerID": 10001967, "GameID": 74546, "HomeRuns": 1},
        ]
    )
    assert len(stats) == 1
    assert int(stats.iloc[0]["sdio_player_id"]) == 10001967
    assert parse_games([{"HomeTeam": "NYY"}]).empty


def test_parse_numeric_sentinels_and_leading_dot() -> None:
    stats = parse_player_game_stats(
        [
            {
                "PlayerID": 42,
                "GameID": 99,
                "PlateAppearances": "--",
                "Hits": True,
                "HomeRuns": "",
                "BattingAverage": ".311",
                "EarnedRunAverage": ".---",
                "InningsPitchedDecimal": "6.2",
            }
        ]
    )
    row = stats.iloc[0]
    assert pd.isna(row["pa"])
    assert pd.isna(row["hits"])
    assert pd.isna(row["hr"])
    assert row["avg"] == pytest.approx(0.311)
    assert pd.isna(row["era"])
    assert row["ip"] == pytest.approx(6.2)


def test_alternate_mlbam_keys_join_lahman() -> None:
    players = parse_players(
        [{"PlayerID": 7, "FirstName": "Alt", "LastName": "Id", "MlbID": 592450}]
    )
    people = pd.DataFrame({"playerID": ["judgeaa01"], "mlbID": [592450], "bbrefID": ["judgeaa01"]})
    joined = attach_lahman_aliases(players, people)
    assert joined.iloc[0]["lahman_player_id"] == "judgeaa01"


def test_attach_team_aliases_maps_oak_ath_and_latest_mia() -> None:
    team_map = load_team_map(TEAM_MAP)
    teams = pd.DataFrame(
        {
            "sdio_team_id": [11, 12, 13],
            "sdio_abbr": ["OAK", "ATH", "MIA"],
        }
    )
    mapped = attach_team_aliases(teams, team_map).set_index("sdio_abbr")
    assert mapped.loc["OAK", "lahman_team_id"] == "OAK"
    assert mapped.loc["ATH", "lahman_team_id"] == "ATH"
    assert int(mapped.loc["OAK", "mlb_team_id"]) == 133
    assert int(mapped.loc["ATH", "mlb_team_id"]) == 133
    assert mapped.loc["MIA", "lahman_team_id"] == "MIA"
    assert int(mapped.loc["MIA", "mlb_team_id"]) == 146


def test_spine_collapses_duplicate_player_game_and_bootstraps_ids() -> None:
    player_game = pd.DataFrame(
        [
            {
                "sdio_player_id": 42,
                "sdio_game_id": 99,
                "sdio_team_id": 7,
                "display_name": "Callup",
                "hr": 1,
                "pa": 4,
            },
            {
                "sdio_player_id": 42,
                "sdio_game_id": 99,
                "sdio_team_id": 7,
                "display_name": "Callup",
                "hr": 2,
                "pa": 4,
            },
        ]
    )
    frames = build_spine_frames(
        teams=pd.DataFrame(),
        players=pd.DataFrame(),
        games=pd.DataFrame(),
        player_game=player_game,
        player_season=pd.DataFrame(),
        as_of_date=AS_OF,
        run_id="edge-run",
        computed_at="2026-08-23T00:00:00Z",
    )
    assert len(frames.player_game_stat) == 1
    assert int(frames.player_game_stat.iloc[0]["hr"]) == 1
    assert frames.player_game_stat.iloc[0]["player_id"] == stable_uuid("player", 42)
    assert frames.player_game_stat.iloc[0]["game_id"] == stable_uuid("game", 99)
    assert list(frames.players["sdio_player_id"]) == [42]
    assert list(frames.teams["sdio_team_id"]) == [7]
    assert list(frames.games["sdio_game_id"]) == [99]
    alias_keys = frames.aliases[["system", "entity_type", "external_id"]]
    assert alias_keys.duplicated().sum() == 0
    assert set(frames.aliases["system"]) <= {"sportsdataio", "mlb", "bbref", "fangraphs", "lahman"}
    assert "fangraphs" not in set(frames.aliases["system"])
    primary = frames.aliases[frames.aliases["is_primary"]]
    assert set(primary["system"]) == {"sportsdataio"}


def test_lake_key_rejects_path_traversal_and_empty_ids() -> None:
    with pytest.raises(ValueError, match="endpoint token"):
        raw_object_key("../teams", AS_OF, "teams.json")
    with pytest.raises(ValueError, match="endpoint token"):
        raw_object_key("teams/secret", AS_OF, "teams.json")
    with pytest.raises(ValueError, match="Invalid raw filename"):
        raw_object_key("teams", AS_OF, "..")
    with pytest.raises(ValueError, match="Invalid raw filename"):
        raw_object_key("teams", AS_OF, "")
    escaped = raw_object_key("teams", AS_OF, "../secret.json")
    assert escaped == f"{RAW_REMOTE_PREFIX}/teams/{AS_OF}/secret.json"
    with pytest.raises(ValueError, match="empty"):
        stable_uuid("player", "  ")


def test_client_sends_key_as_header_not_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.baseball_analytics.sportsdataio._backoff", lambda *_a, **_k: None)
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = [{"TeamID": 31}]
    response.text = "ok"
    session.get.return_value = response
    client = SportsDataIOClient(
        api_key="super-secret-key",
        session=session,
        min_interval=0,
        max_retries=0,
    )
    payload = client.teams()
    assert payload == [{"TeamID": 31}]
    _args, kwargs = session.get.call_args
    url = _args[0]
    params = kwargs.get("params") or {}
    headers = kwargs.get("headers") or {}
    assert headers["Ocp-Apim-Subscription-Key"] == "super-secret-key"
    assert "key" not in params
    assert "super-secret-key" not in url
    assert "?key=" not in url


def test_client_401_does_not_retry_and_hides_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.baseball_analytics.sportsdataio._backoff", lambda *_a, **_k: None)
    session = MagicMock()
    response = MagicMock()
    response.status_code = 401
    response.text = "unauthorized"
    session.get.return_value = response
    client = SportsDataIOClient(
        api_key="super-secret-key",
        session=session,
        min_interval=0,
        max_retries=3,
    )
    with pytest.raises(SportsDataIOError) as excinfo:
        client.teams()
    assert session.get.call_count == 1
    assert excinfo.value.status_code == 401
    assert "super-secret-key" not in str(excinfo.value)


def test_client_retries_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.baseball_analytics.sportsdataio._backoff", lambda *_a, **_k: None)
    busy = MagicMock()
    busy.status_code = 429
    busy.text = "slow down"
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = [{"TeamID": 31}]
    ok.text = "ok"
    session = MagicMock()
    session.get.side_effect = [busy, ok]
    client = SportsDataIOClient(
        api_key="test-key",
        session=session,
        min_interval=0,
        max_retries=1,
    )
    assert client.teams() == [{"TeamID": 31}]
    assert session.get.call_count == 2


def test_client_get_without_key_raises_missing() -> None:
    client = SportsDataIOClient(api_key=None, environ={}, min_interval=0)
    with pytest.raises(MissingApiKeyError):
        client.teams()


def test_date_endpoints_use_month_abbrev_token() -> None:
    seen: list[str] = []

    def fetcher(path: str, params: dict) -> list:
        seen.append(path)
        return []

    client = SportsDataIOClient(api_key="test-key", fetcher=fetcher, min_interval=0)
    client.games_by_date("2024-07-31")
    client.player_game_stats_by_date("2017-09-01")
    assert seen[0].endswith("/GamesByDate/2024-JUL-31")
    assert seen[1].endswith("/PlayerGameStatsByDate/2017-SEP-01")


def test_seasons_from_settings_prefers_env_then_yaml() -> None:
    assert seasons_from_settings({}, "2026-08-23", environ={}) == [2026]
    assert seasons_from_settings(
        {"sportsdataio": {"seasons": [2024, 2025]}},
        "2026-08-23",
        environ={},
    ) == [2024, 2025]
    assert seasons_from_settings(
        {"sportsdataio": {"seasons": [2024]}},
        "2026-08-23",
        environ={"SPORTSDATAIO_SEASONS": "2023, 2024"},
    ) == [2023, 2024]


def test_resolve_as_of_date_prefers_explicit_then_local(tmp_path: Path) -> None:
    write_raw_payload(
        [{"TeamID": 1}],
        endpoint=ENDPOINT_TEAMS,
        as_of_date="2024-07-01",
        filename="teams.json",
        raw_dir=tmp_path,
    )
    assert resolve_as_of_date(tmp_path, as_of_date="2026-08-23") == "2026-08-23"
    assert (
        resolve_as_of_date(tmp_path, environ={"ARTIFACTS_AS_OF_DATE": "2024-07-01"})
        == "2024-07-01"
    )
    assert (
        resolve_as_of_date(tmp_path, environ={"ARTIFACTS_AS_OF_DATE": "1999-01-01"})
        == "2024-07-01"
    )
    empty = tmp_path / "empty-raw"
    empty.mkdir()
    assert (
        resolve_as_of_date(empty, environ={"ARTIFACTS_AS_OF_DATE": "2024-07-04"})
        == "2024-07-04"
    )
