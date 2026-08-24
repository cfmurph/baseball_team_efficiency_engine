"""Extract, parse, and Lahman-join tests for the MLB Stats API ingest (#108)."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from pipeline.extract import pull_mlb_stats as pull_mod
from pipeline.transform.build_warehouse import insert_mlb_stats_tables
from src.baseball_analytics.mlb_stats import (
    ENDPOINT_TEAMS,
    MlbFrames,
    MlbStatsClient,
    MlbStatsError,
    RAW_REMOTE_PREFIX,
    discover_as_of_dates,
    join_mlb_player_ids,
    join_mlb_team_ids,
    load_mlb_frames,
    load_team_map,
    local_raw_path,
    parse_player_stats,
    parse_schedule,
    parse_standings,
    parse_team_stats,
    parse_teams,
    pull_majors_feeds,
    raw_object_key,
    read_raw_payload,
    write_raw_payload,
    _merge_player_seasons,
)
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.storage import FileBackend, default_as_of_date

FIXTURES = Path(__file__).parent / "fixtures" / "mlb_stats"
TEAM_MAP = Path(__file__).resolve().parents[1] / "data" / "crosswalks" / "mlb_team_map.csv"
AS_OF = "2026-08-23"


def _payload(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _land_fixtures(raw_dir: Path, as_of: str = AS_OF, backend: FileBackend | None = None) -> None:
    mapping = {
        ("teams", "teams.json"): "teams.json",
        ("standings", "standings_2024.json"): "standings_2024.json",
        ("team_hitting", "team_hitting_2024.json"): "team_hitting_2024.json",
        ("team_pitching", "team_pitching_2024.json"): "team_pitching_2024.json",
        ("player_hitting", "player_hitting_2024.json"): "player_hitting_2024.json",
        ("player_pitching", "player_pitching_2024.json"): "player_pitching_2024.json",
        ("schedule", "schedule_2024.json"): "schedule_2024.json",
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
        {"as_of_date": as_of, "seasons": [2024], "ok": True, "endpoints": []},
        endpoint="extract_report",
        as_of_date=as_of,
        filename="extract_report.json",
        raw_dir=raw_dir,
        backend=backend,
    )


def test_raw_object_key_matches_locked_layout() -> None:
    key = raw_object_key("player_hitting", AS_OF, "player_hitting_2024.json")
    assert key == f"{RAW_REMOTE_PREFIX}/player_hitting/{AS_OF}/player_hitting_2024.json"
    local = local_raw_path("data/raw", "teams", AS_OF, "teams.json")
    assert local.as_posix().endswith(f"data/raw/mlb_stats/teams/{AS_OF}/teams.json")


def test_parse_teams_keeps_majors_ids() -> None:
    teams = parse_teams(_payload("teams.json"))
    assert set(teams["mlb_team_id"]) == {147, 133}
    yankees = teams.set_index("mlb_team_id").loc[147]
    assert yankees["mlb_abbr"] == "NYY"
    assert yankees["league_id"] == 103


def test_parse_standings_and_team_stats() -> None:
    standings = parse_standings(_payload("standings_2024.json"))
    assert standings.iloc[0]["wins"] == 94
    assert standings.iloc[0]["winning_pct"] == pytest.approx(0.580)
    hitting = parse_team_stats(_payload("team_hitting_2024.json"), "hitting")
    assert hitting.iloc[0]["batting_hr"] == 237
    pitching = parse_team_stats(_payload("team_pitching_2024.json"), "pitching")
    assert pitching.iloc[0]["era"] == pytest.approx(3.74)
    assert pitching.iloc[0]["ip"] == pytest.approx(1446.0)


def test_parse_player_stats_and_schedule() -> None:
    hitting = parse_player_stats(_payload("player_hitting_2024.json"), "hitting")
    judge = hitting.set_index("mlb_player_id").loc[592450]
    assert judge["hr"] == 58
    assert judge["player_name"] == "Aaron Judge"
    games = parse_schedule(_payload("schedule_2024.json"))
    assert len(games) == 1
    assert int(games.iloc[0]["game_pk"]) == 745460
    assert int(games.iloc[0]["home_score"]) == 6
    assert int(games.iloc[0]["away_mlb_team_id"]) == 113


def test_join_mlb_team_to_lahman_is_year_aware() -> None:
    team_map = load_team_map(TEAM_MAP)
    rows = pd.DataFrame(
        {
            "mlb_team_id": [147, 133, 133, 999],
            "season_year": [2024, 2024, 2026, 2024],
        }
    )
    joined = join_mlb_team_ids(rows, team_map)
    by_key = joined.set_index(["mlb_team_id", "season_year"])["lahman_team_id"]
    assert by_key.loc[(147, 2024)] == "NYA"
    assert by_key.loc[(133, 2024)] == "OAK"
    assert by_key.loc[(133, 2026)] == "ATH"
    assert pd.isna(by_key.loc[(999, 2024)])


def test_join_mlb_player_to_lahman_via_people_mlbid() -> None:
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01", "ohtansh01"],
            "mlbID": [592450, 660271],
        }
    )
    players = pd.DataFrame({"mlb_player_id": [592450, 123], "player_name": ["Aaron Judge", "Unknown"]})
    joined = join_mlb_player_ids(players, people)
    assert joined.set_index("mlb_player_id").loc[592450, "lahman_player_id"] == "judgeaa01"
    assert pd.isna(joined.set_index("mlb_player_id").loc[123, "lahman_player_id"])


def test_join_leaves_null_when_people_has_no_mlbid() -> None:
    people = pd.DataFrame({"playerID": ["judgeaa01"], "bbrefID": ["judgeaa01"]})
    players = pd.DataFrame({"mlb_player_id": [592450], "player_name": ["Aaron Judge"]})
    joined = join_mlb_player_ids(players, people)
    assert pd.isna(joined.iloc[0]["lahman_player_id"])


def test_extract_writes_local_and_file_uri(tmp_path: Path) -> None:
    raw_dir = tmp_path / "data" / "raw"
    lake = tmp_path / "lake"
    backend = FileBackend(lake)

    def fetcher(path: str, params: dict) -> dict:
        if path.endswith("/teams"):
            return _payload("teams.json")
        if path.endswith("/standings"):
            return _payload("standings_2024.json")
        if path.endswith("/teams/stats") and params.get("group") == "hitting":
            return _payload("team_hitting_2024.json")
        if path.endswith("/teams/stats"):
            return _payload("team_pitching_2024.json")
        if path.endswith("/stats") and params.get("group") == "hitting":
            return _payload("player_hitting_2024.json")
        if path.endswith("/stats"):
            return _payload("player_pitching_2024.json")
        if path.endswith("/schedule"):
            return _payload("schedule_2024.json")
        raise MlbStatsError(f"unexpected path {path}")

    client = MlbStatsClient(fetcher=fetcher, min_interval=0)
    report = pull_majors_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
        backend=backend,
        schedule_mode="season",
    )
    assert report.ok
    local_teams = local_raw_path(raw_dir, "teams", AS_OF, "teams.json")
    assert local_teams.is_file()
    remote_teams = lake / raw_object_key("teams", AS_OF, "teams.json")
    assert remote_teams.is_file()
    assert json.loads(local_teams.read_text()) == json.loads(remote_teams.read_text())
    assert (lake / raw_object_key("extract_report", AS_OF, "extract_report.json")).is_file()
    assert discover_as_of_dates(raw_dir) == [AS_OF]


def test_extract_same_date_overwrite_is_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    write_raw_payload(
        {"teams": [{"id": 1}]},
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
    assert len(landed["teams"]) == 2


def test_extract_soft_fails_on_api_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    def fetcher(path: str, params: dict) -> dict:
        if path.endswith("/teams"):
            return _payload("teams.json")
        raise MlbStatsError("HTTP 503 from statsapi", status_code=503, url=path)

    client = MlbStatsClient(fetcher=fetcher, min_interval=0)
    report = pull_majors_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
    )
    assert report.ok is False
    assert any(item.endpoint == "teams" and item.ok for item in report.endpoints)
    assert any(item.endpoint == "player_hitting" and not item.ok for item in report.endpoints)
    assert local_raw_path(raw_dir, "teams", AS_OF, "teams.json").is_file()


def test_cli_soft_fail_exits_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "raw_dir: raw\nartifacts_uri: ''\nartifacts_dir: artifacts\nmlb_stats: {}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def boom(*_args, **_kwargs):
        raise RuntimeError("statsapi down")

    monkeypatch.setattr(pull_mod, "pull_majors_feeds", boom)
    result = CliRunner().invoke(
        pull_mod.app,
        ["--config-path", str(settings_path), "--as-of-date", AS_OF],
    )
    assert result.exit_code == 0
    report = json.loads(
        (tmp_path / "raw" / "mlb_stats" / "extract_report" / AS_OF / "extract_report.json").read_text()
    )
    assert report["ok"] is False
    assert report["soft_fail"] is True


def test_warehouse_builds_without_stats_api(tmp_path: Path) -> None:
    frames = load_mlb_frames(tmp_path / "missing-raw", as_of_date=AS_OF)
    assert frames.empty
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    loaded = insert_mlb_stats_tables(con, frames)
    assert loaded == {}
    count = con.execute("SELECT COUNT(*) FROM fact_mlb_team_season").fetchone()[0]
    assert count == 0


def test_warehouse_loads_stats_api_joins_and_skips_war(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _land_fixtures(raw_dir)
    people = pd.DataFrame(
        {
            "playerID": ["judgeaa01", "ohtansh01"],
            "nameFirst": ["Aaron", "Shohei"],
            "nameLast": ["Judge", "Ohtani"],
            "mlbID": [592450, 660271],
        }
    )
    frames = load_mlb_frames(
        raw_dir,
        as_of_date=AS_OF,
        people=people,
        team_map_path=TEAM_MAP,
    )
    assert not frames.empty
    yankees = frames.team_season.set_index("mlb_team_id").loc[147]
    assert yankees["lahman_team_id"] == "NYA"
    assert yankees["wins"] == 94
    judge = frames.player_season.set_index("mlb_player_id").loc[592450]
    assert judge["lahman_player_id"] == "judgeaa01"
    assert judge["lahman_team_id"] == "NYA"
    ohtani = frames.player_season.set_index("mlb_player_id").loc[660271]
    assert ohtani["player_type"] == "both"
    assert ohtani["lahman_team_id"] == "LAN"
    game = frames.games.iloc[0]
    assert game["home_lahman_team_id"] == "PIT"
    assert game["away_lahman_team_id"] == "CIN"
    for frame in (frames.team_season, frames.player_season, frames.games):
        assert not any("war" in col.lower() for col in frame.columns)

    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    con.execute(
        "INSERT INTO fact_player_season (player_id, season_key, team_id, player_war, war_source) "
        "VALUES ('judgeaa01', 2024, 'NYA', 10.8, 'real')"
    )
    loaded = insert_mlb_stats_tables(con, frames)
    assert loaded["fact_mlb_team_season"] >= 1
    assert loaded["fact_mlb_player_season"] >= 1
    assert loaded["fact_mlb_game"] == 1
    war = con.execute(
        "SELECT player_war, war_source FROM fact_player_season WHERE player_id = 'judgeaa01'"
    ).fetchone()
    assert war == (10.8, "real")
    mlb_cols = [r[1] for r in con.execute("PRAGMA table_info('fact_mlb_player_season')").fetchall()]
    assert "player_war" not in mlb_cols
    assert "war_source" not in mlb_cols


def test_load_mlb_frames_reads_file_uri_when_local_missing(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    backend = FileBackend(lake)
    _land_fixtures(tmp_path / "seed", backend=backend)
    empty_local = tmp_path / "empty-raw"
    empty_local.mkdir()
    frames = load_mlb_frames(
        empty_local,
        as_of_date=AS_OF,
        team_map_path=TEAM_MAP,
        backend=backend,
    )
    assert not frames.team_season.empty
    assert int(frames.team_season.iloc[0]["mlb_team_id"]) == 147


def test_default_as_of_date_env_is_shared_with_lake() -> None:
    assert default_as_of_date(environ={"ARTIFACTS_AS_OF_DATE": "2024-07-04"}) == "2024-07-04"


def test_insert_rejects_war_column() -> None:
    frames = MlbFrames(
        as_of_date=AS_OF,
        team_season=pd.DataFrame(
            {"mlb_team_id": [147], "season_year": [2024], "player_war": [5.0]}
        ),
    )
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    with pytest.raises(ValueError, match="must not write WAR"):
        insert_mlb_stats_tables(con, frames)


def test_parse_schedule_drops_missing_pk_and_keeps_postponed_null_scores() -> None:
    payload = {
        "dates": [
            {
                "date": "2024-08-23",
                "games": [
                    {
                        "season": "2024",
                        "officialDate": "2024-08-23",
                        "status": {"detailedState": "Cancelled"},
                        "teams": {
                            "away": {"team": {"id": 113}},
                            "home": {"team": {"id": 134}},
                        },
                    },
                    {
                        "gamePk": 745461,
                        "season": "2024",
                        "officialDate": "2024-08-23",
                        "status": {"detailedState": "Postponed", "abstractGameState": "Preview"},
                        "venue": {"name": "PNC Park"},
                        "teams": {
                            "away": {"team": {"id": 113}},
                            "home": {"team": {"id": 134}},
                        },
                    },
                ],
            }
        ]
    }
    games = parse_schedule(payload)
    assert list(games["game_pk"]) == [745461]
    assert games.iloc[0]["status"] == "Postponed"
    assert pd.isna(games.iloc[0]["home_score"])
    assert pd.isna(games.iloc[0]["away_score"])
    assert int(games.iloc[0]["home_mlb_team_id"]) == 134


def test_parse_player_stats_drops_missing_ids_and_sentinels() -> None:
    payload = {
        "stats": [
            {
                "splits": [
                    {
                        "season": "2024",
                        "player": {"fullName": "No Id"},
                        "team": {"id": 147},
                        "stat": {"plateAppearances": 10},
                    },
                    {
                        "season": "2024",
                        "player": {"id": 592450, "fullName": "Aaron Judge"},
                        "team": {"id": 147, "name": "Yankees"},
                        "stat": {
                            "gamesPlayed": 1,
                            "plateAppearances": "--",
                            "hits": True,
                            "avg": ".311",
                            "homeRuns": "",
                        },
                    },
                ]
            }
        ]
    }
    hitting = parse_player_stats(payload, "hitting")
    assert list(hitting["mlb_player_id"]) == [592450]
    row = hitting.iloc[0]
    assert pd.isna(row["pa"])
    assert pd.isna(row["hits"])
    assert row["avg"] == pytest.approx(0.311)
    assert pd.isna(row["hr"])


def test_merge_player_seasons_labels_pitcher_batter_and_two_way() -> None:
    hitting = pd.DataFrame(
        {
            "mlb_player_id": [1, 3],
            "season_year": [2024, 2024],
            "mlb_team_id": [147, 119],
            "player_name": ["Bat", "TwoWay"],
            "pa": [500, 400],
            "hr": [20, 30],
        }
    )
    pitching = pd.DataFrame(
        {
            "mlb_player_id": [2, 3],
            "season_year": [2024, 2024],
            "mlb_team_id": [143, 119],
            "player_name": ["Arm", "TwoWay"],
            "ip": [180.0, 12.0],
            "era": [3.20, 2.25],
            "pitching_so": [200, 16],
            "pitching_bb": [40, 3],
        }
    )
    merged = _merge_player_seasons([hitting], [pitching]).set_index("mlb_player_id")
    assert merged.loc[1, "player_type"] == "batter"
    assert merged.loc[2, "player_type"] == "pitcher"
    assert merged.loc[3, "player_type"] == "both"
    assert merged.loc[2, "era"] == pytest.approx(3.20)
    pitcher_only = _merge_player_seasons([], [pitching])
    assert set(pitcher_only["player_type"]) == {"pitcher"}
    batter_only = _merge_player_seasons([hitting], [])
    assert set(batter_only["player_type"]) == {"batter"}
