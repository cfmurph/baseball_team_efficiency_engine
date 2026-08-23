"""Extract, alias, spine, and soft-fail tests for SportsDataIO Phase 0 (#128)."""
from __future__ import annotations

import json
from pathlib import Path

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
    SportsDataIOClient,
    SportsDataIOError,
    attach_lahman_aliases,
    discover_as_of_dates,
    load_sdio_frames,
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
    sdio_date_token,
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


def test_raw_object_key_matches_locked_layout() -> None:
    key = raw_object_key("player_game_stats", AS_OF, "player_game_stats_2026-08-23.json")
    assert key == f"{RAW_REMOTE_PREFIX}/player_game_stats/{AS_OF}/player_game_stats_2026-08-23.json"
    local = local_raw_path("data/raw", "teams", AS_OF, "teams.json")
    assert local.as_posix().endswith(f"data/raw/sportsdataio/teams/{AS_OF}/teams.json")


def test_sdio_date_token_uses_month_abbrev() -> None:
    assert sdio_date_token("2024-07-31") == "2024-JUL-31"
    assert sdio_date_token("2017-09-01") == "2017-SEP-01"


def test_stable_uuid_is_deterministic() -> None:
    first = stable_uuid("player", 10001967)
    second = stable_uuid("player", 10001967)
    assert first == second
    assert first != stable_uuid("team", 10001967)


def test_resolve_api_key_reads_env_only() -> None:
    assert resolve_api_key(environ={}) is None
    assert resolve_api_key(environ={API_KEY_ENV: "  secret-key  "}) == "secret-key"
    assert resolve_api_key("cli-key", environ={API_KEY_ENV: "env-key"}) == "cli-key"


def test_parse_teams_and_players() -> None:
    teams = parse_teams(_payload("teams.json"))
    assert set(teams["sdio_team_id"]) == {31, 20}
    yankees = teams.set_index("sdio_team_id").loc[31]
    assert yankees["sdio_abbr"] == "NYY"
    players = parse_players(_payload("players.json"))
    judge = players.set_index("sdio_player_id").loc[10001967]
    assert judge["display_name"] == "Aaron Judge"
    assert int(judge["mlb_player_id"]) == 592450


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
    landed = read_raw_payload(
        endpoint="extract_report",
        as_of_date=AS_OF,
        filename="extract_report.json",
        raw_dir=raw_dir,
    )
    assert landed["ok"] is False
    assert landed["soft_fail"] is True


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
    assert "SPORTSDATAIO_API_KEY" in (report.get("error") or "")


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


def test_sdio_probe_workflow_is_dispatch_only() -> None:
    text = Path(".github/workflows/sdio-probe.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "pull_request:" not in text
    assert "SPORTSDATAIO_API_KEY: ${{ secrets.SPORTSDATAIO_API_KEY }}" in text
    assert "Ocp-Apim-Subscription-Key" in text
    assert "SPORTSDATAIO_API_KEY missing; skipping probe (soft-fail)" in text
    assert "sys.exit(0)" in text
    assert "/v3/mlb/scores/json/Teams" in text
    assert "HTTPError" in text
    assert "http_status={int(exc.code)}" in text
    assert "print(key" not in text
    assert "?key=" not in text
    nightly = Path(".github/workflows/nightly-refresh.yml").read_text(encoding="utf-8")
    assert "SPORTSDATAIO_API_KEY: ${{ secrets.SPORTSDATAIO_API_KEY }}" in nightly
    smoke = Path(".github/workflows/ci-smoke.yml").read_text(encoding="utf-8")
    assert "secrets.SPORTSDATAIO_API_KEY" not in smoke


def test_warehouse_ddl_has_no_forked_stat_tables() -> None:
    assert "CREATE OR REPLACE TABLE fantasy_" not in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE scout_" not in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE external_id_alias" in WAREHOUSE_DDL
    assert "CREATE OR REPLACE TABLE player_game_stat" in WAREHOUSE_DDL
    assert "PRIMARY KEY (player_id, game_id)" in WAREHOUSE_DDL
