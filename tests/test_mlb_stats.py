"""Extract, parse, and Lahman-join tests for the MLB Stats API ingest (#108)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest

from pipeline.extract import pull_mlb_stats as pull_mod
from pipeline.transform.build_warehouse import insert_mlb_stats_tables
from src.baseball_analytics.mlb_stats import (
    ENDPOINT_TEAMS,
    EndpointResult,
    ExtractReport,
    MlbFrames,
    MlbStatsClient,
    MlbStatsError,
    RAW_REMOTE_PREFIX,
    client_from_settings,
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

@pytest.mark.unit
def test_raw_object_key_matches_locked_layout() -> None:
    key = raw_object_key("player_hitting", AS_OF, "player_hitting_2024.json")
    assert key == f"{RAW_REMOTE_PREFIX}/player_hitting/{AS_OF}/player_hitting_2024.json"
    local = local_raw_path("data/raw", "teams", AS_OF, "teams.json")
    assert local.as_posix().endswith(f"data/raw/mlb_stats/teams/{AS_OF}/teams.json")

@pytest.mark.unit
def test_parse_teams_keeps_majors_ids() -> None:
    teams = parse_teams(_payload("teams.json"))
    assert set(teams["mlb_team_id"]) == {147, 133}
    yankees = teams.set_index("mlb_team_id").loc[147]
    assert yankees["mlb_abbr"] == "NYY"
    assert yankees["league_id"] == 103

@pytest.mark.unit
def test_parse_standings_and_team_stats() -> None:
    standings = parse_standings(_payload("standings_2024.json"))
    assert standings.iloc[0]["wins"] == 94
    assert standings.iloc[0]["winning_pct"] == pytest.approx(0.580)
    hitting = parse_team_stats(_payload("team_hitting_2024.json"), "hitting")
    assert hitting.iloc[0]["batting_hr"] == 237
    pitching = parse_team_stats(_payload("team_pitching_2024.json"), "pitching")
    assert pitching.iloc[0]["era"] == pytest.approx(3.74)
    assert pitching.iloc[0]["ip"] == pytest.approx(1446.0)

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
def test_join_leaves_null_when_people_has_no_mlbid() -> None:
    people = pd.DataFrame({"playerID": ["judgeaa01"], "bbrefID": ["judgeaa01"]})
    players = pd.DataFrame({"mlb_player_id": [592450], "player_name": ["Aaron Judge"]})
    joined = join_mlb_player_ids(players, people)
    assert pd.isna(joined.iloc[0]["lahman_player_id"])

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.integration
def test_warehouse_builds_without_stats_api(tmp_path: Path) -> None:
    frames = load_mlb_frames(tmp_path / "missing-raw", as_of_date=AS_OF)
    assert frames.empty
    con = duckdb.connect(":memory:")
    con.execute(WAREHOUSE_DDL)
    loaded = insert_mlb_stats_tables(con, frames)
    assert loaded == {}
    count = con.execute("SELECT COUNT(*) FROM fact_mlb_team_season").fetchone()[0]
    assert count == 0

@pytest.mark.integration
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

@pytest.mark.integration
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

@pytest.mark.unit
def test_warehouse_ddl_keeps_war_on_lahman_facts_only() -> None:
    """No dual-write WAR: MLB Stats facts have no war / war_source columns."""
    lahman_player = WAREHOUSE_DDL.split("CREATE OR REPLACE TABLE fact_player_season")[1].split(
        "CREATE OR REPLACE TABLE"
    )[0]
    assert "war_source" in lahman_player
    mlb_team = WAREHOUSE_DDL.split("CREATE OR REPLACE TABLE fact_mlb_team_season")[1].split(
        "CREATE OR REPLACE TABLE"
    )[0]
    mlb_player = WAREHOUSE_DDL.split("CREATE OR REPLACE TABLE fact_mlb_player_season")[1].split(
        "CREATE OR REPLACE TABLE"
    )[0]
    assert "No WAR columns" in WAREHOUSE_DDL
    for section in (mlb_team, mlb_player):
        assert "war_source" not in section
        assert "player_war" not in section
        assert "\n    war " not in section.lower()

@pytest.mark.unit
def test_default_as_of_date_env_is_shared_with_lake() -> None:
    assert default_as_of_date(environ={"ARTIFACTS_AS_OF_DATE": "2024-07-04"}) == "2024-07-04"

@pytest.mark.integration
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_parse_snake_case_library_dumps() -> None:
    teams = parse_teams(
        {
            "teams": [
                {
                    "id": 147,
                    "name": "New York Yankees",
                    "abbreviation": "NYY",
                    "team_name": "Yankees",
                    "location_name": "New York",
                    "active": True,
                    "league": {"id": 103, "name": "American League"},
                    "division": {"id": 201, "name": "American League East"},
                    "sport": {"id": 1},
                }
            ]
        }
    )
    assert teams.iloc[0]["mlb_abbr"] == "NYY"
    assert teams.iloc[0]["mlb_team_name"] == "Yankees"
    standings = parse_standings(
        {
            "records": [
                {
                    "team_records": [
                        {
                            "team": {"id": 147, "name": "Yankees"},
                            "season": "2024",
                            "wins": 94,
                            "losses": 68,
                            "games_played": 162,
                            "runs_scored": 815,
                            "runs_allowed": 668,
                            "run_differential": 147,
                            "winning_percentage": 0.58,
                            "division_rank": "1",
                            "league_rank": "1",
                        }
                    ]
                }
            ]
        }
    )
    assert standings.iloc[0]["wins"] == 94
    assert standings.iloc[0]["winning_pct"] == pytest.approx(0.58)
    hitting = parse_player_stats(
        {
            "stats": [
                {
                    "splits": [
                        {
                            "season": "2024",
                            "player": {"id": 592450, "full_name": "Aaron Judge"},
                            "team": {"id": 147, "name": "Yankees"},
                            "stat": {
                                "games_played": 158,
                                "plate_appearances": 704,
                                "at_bats": 559,
                                "hits": 180,
                                "home_runs": 58,
                                "base_on_balls": 133,
                                "strike_outs": 171,
                                "avg": 0.322,
                                "obp": 0.458,
                                "slg": 0.701,
                                "ops": 1.159,
                            },
                        }
                    ]
                }
            ]
        },
        "hitting",
    )
    judge = hitting.set_index("mlb_player_id").loc[592450]
    assert judge["hr"] == 58
    assert judge["player_name"] == "Aaron Judge"


@pytest.mark.unit
def test_parse_player_stats_does_not_land_drs_oaa_uzr() -> None:
    hitting = parse_player_stats(
        {
            "stats": [
                {
                    "splits": [
                        {
                            "season": "2024",
                            "player": {"id": 1, "full_name": "X"},
                            "team": {"id": 147},
                            "stat": {
                                "games_played": 10,
                                "home_runs": 2,
                                "drs": 12,
                                "oaa": 5,
                                "uzr": 3,
                                "outs_above_average": 4,
                            },
                        }
                    ]
                }
            ]
        },
        "hitting",
    )
    assert list(hitting["mlb_player_id"]) == [1]
    assert hitting.iloc[0]["hr"] == 2
    for banned in ("drs", "oaa", "uzr", "outs_above_average", "war"):
        assert banned not in hitting.columns


@pytest.mark.unit
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


class _Dump:
    """Stand-in for a Pydantic model so tests never construct live Mlb HTTP."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.id = payload.get("id")

    def model_dump(self, **_kwargs) -> dict:
        return dict(self._payload)


@pytest.mark.unit
def test_client_uses_library_get_stats_and_dumps_snake_case() -> None:
    mlb = MagicMock()
    mlb.get_stats.return_value = {
        "hitting": {
            "season": _Dump(
                {
                    "splits": [
                        {
                            "season": "2024",
                            "player": {"id": 592450, "full_name": "Aaron Judge"},
                            "team": {"id": 147, "name": "Yankees"},
                            "stat": {"games_played": 158, "home_runs": 58, "plate_appearances": 704},
                        }
                    ]
                }
            )
        }
    }
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    payload = client.player_stats(2024, "hitting")
    mlb.get_stats.assert_called_once()
    kwargs = mlb.get_stats.call_args.kwargs
    assert kwargs["season"] == 2024
    assert kwargs["playerPool"] == "all"
    hitting = parse_player_stats(payload, "hitting")
    assert int(hitting.iloc[0]["mlb_player_id"]) == 592450
    assert hitting.iloc[0]["hr"] == 58
    assert hitting.iloc[0]["player_name"] == "Aaron Judge"


@pytest.mark.unit
def test_client_uses_library_get_team_stats_and_schedule() -> None:
    mlb = MagicMock()
    mlb.get_teams.return_value = [_Dump({"id": 147, "abbreviation": "NYY", "name": "Yankees"})]
    mlb.get_team_stats.return_value = {
        "hitting": {
            "season": _Dump(
                {
                    "splits": [
                        {
                            "season": "2024",
                            "team": {"id": 147, "name": "Yankees"},
                            "stat": {"home_runs": 237, "games_played": 162},
                        }
                    ]
                }
            )
        }
    }
    mlb.get_schedule.return_value = _Dump(
        {
            "dates": [
                {
                    "date": "2024-08-23",
                    "games": [
                        {
                            "game_pk": 745460,
                            "season": "2024",
                            "official_date": "2024-08-23",
                            "status": {"detailed_state": "Final"},
                            "venue": {"name": "PNC Park"},
                            "teams": {
                                "away": {"team": {"id": 113}, "score": 5, "league_record": {"wins": 62, "losses": 67}},
                                "home": {"team": {"id": 134}, "score": 6, "league_record": {"wins": 61, "losses": 67}},
                            },
                        }
                    ],
                }
            ]
        }
    )
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    teams = parse_teams(client.teams())
    hitting = parse_team_stats(client.team_stats(2024, "hitting"), "hitting")
    games = parse_schedule(client.schedule(season=2024))
    mlb.get_teams.assert_called()
    mlb.get_team_stats.assert_called_once()
    mlb.get_schedule.assert_called_once()
    assert list(teams["mlb_team_id"]) == [147]
    assert hitting.iloc[0]["batting_hr"] == 237
    assert int(games.iloc[0]["game_pk"]) == 745460
    assert int(games.iloc[0]["home_score"]) == 6


@pytest.mark.unit
def test_client_maps_mlb_http_error() -> None:
    from mlbstatsapi import MlbHttpError

    mlb = MagicMock()
    mlb.get_teams.side_effect = MlbHttpError(
        400,
        "bad request",
        url="https://statsapi.mlb.com/api/v1/teams",
    )
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    with pytest.raises(MlbStatsError) as excinfo:
        client.teams()
    assert excinfo.value.status_code == 400
    mlb.get_teams.assert_called_once()


@pytest.mark.unit
def test_client_maps_mlb_timeout() -> None:
    from mlbstatsapi import MlbTimeoutError

    mlb = MagicMock()
    mlb.get_stats.side_effect = MlbTimeoutError("Request failed")
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    with pytest.raises(MlbStatsError, match="timeout"):
        client.player_stats(2024, "hitting")


@pytest.mark.unit
def test_fetcher_path_does_not_construct_mlb(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("Mlb must not be constructed when fetcher is set")

    monkeypatch.setattr("src.baseball_analytics.mlb_stats.Mlb", boom)
    client = MlbStatsClient(fetcher=lambda _path, _params: {"teams": []}, min_interval=0)
    assert client.teams() == {"teams": []}


@pytest.mark.unit
def test_team_stats_skips_one_failed_team() -> None:
    from mlbstatsapi import MlbHttpError

    mlb = MagicMock()
    mlb.get_teams.return_value = [
        _Dump({"id": 147, "name": "Yankees"}),
        _Dump({"id": 133, "name": "Athletics"}),
    ]

    def team_stats(team_id, stats, groups, **_kwargs):
        if team_id == 147:
            raise MlbHttpError(503, "unavailable", url="/teams/147/stats")
        return {
            "hitting": {
                "season": _Dump(
                    {
                        "splits": [
                            {
                                "season": "2024",
                                "team": {"id": 133, "name": "Athletics"},
                                "stat": {"home_runs": 180, "games_played": 162},
                            }
                        ]
                    }
                )
            }
        }

    mlb.get_team_stats.side_effect = team_stats
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    hitting = parse_team_stats(client.team_stats(2024, "hitting"), "hitting")
    assert list(hitting["mlb_team_id"]) == [133]
    assert hitting.iloc[0]["batting_hr"] == 180


@pytest.mark.unit
def test_client_from_settings_uses_library_host_and_strict_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: dict = {}

    def fake_mlb(**kwargs):
        constructed.update(kwargs)
        return MagicMock()

    monkeypatch.setattr("src.baseball_analytics.mlb_stats.Mlb", fake_mlb)
    client = client_from_settings({"mlb_stats": {"base_url": "https://statsapi.mlb.com"}})
    assert constructed["hostname"] == "statsapi.mlb.com"
    assert constructed["strict_http"] is True
    client.close()


@pytest.mark.integration
def test_extract_date_schedule_lands_as_of_filename(tmp_path: Path) -> None:
    seen: list[tuple[str, dict]] = []

    def fetcher(path: str, params: dict) -> dict:
        seen.append((path, dict(params)))
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

    raw_dir = tmp_path / "raw"
    client = MlbStatsClient(fetcher=fetcher, min_interval=0)
    report = pull_majors_feeds(
        raw_dir=raw_dir,
        as_of_date=AS_OF,
        seasons=[2024],
        client=client,
        schedule_mode="date",
    )
    assert report.ok
    schedule_calls = [params for path, params in seen if path.endswith("/schedule")]
    assert schedule_calls
    assert schedule_calls[0].get("date") == AS_OF
    assert schedule_calls[0].get("season") is None
    assert local_raw_path(raw_dir, "schedule", AS_OF, f"schedule_{AS_OF}.json").is_file()
    assert not local_raw_path(raw_dir, "schedule", AS_OF, "schedule_2024.json").is_file()


@pytest.mark.unit
def test_parse_teams_skips_non_majors_sport() -> None:
    teams = parse_teams(
        {
            "teams": [
                {
                    "id": 147,
                    "abbreviation": "NYY",
                    "name": "New York Yankees",
                    "sport": {"id": 1},
                },
                {
                    "id": 564,
                    "abbreviation": "SWB",
                    "name": "Scranton/Wilkes-Barre RailRiders",
                    "sport": {"id": 11},
                },
            ]
        }
    )
    assert list(teams["mlb_team_id"]) == [147]


@pytest.mark.unit
def test_parse_player_stats_accepts_stats_object_not_list() -> None:
    hitting = parse_player_stats(
        {
            "stats": {
                "splits": [
                    {
                        "season": "2024",
                        "player": {"id": 592450, "fullName": "Aaron Judge"},
                        "team": {"id": 147, "name": "Yankees"},
                        "stat": {"homeRuns": 58, "gamesPlayed": 158},
                    }
                ]
            }
        },
        "hitting",
    )
    assert list(hitting["mlb_player_id"]) == [592450]
    assert hitting.iloc[0]["hr"] == 58


@pytest.mark.unit
def test_client_standings_uses_library_dump() -> None:
    mlb = MagicMock()
    mlb.get_standings.return_value = [
        _Dump(
            {
                "team_records": [
                    {
                        "team": {"id": 147, "name": "Yankees"},
                        "season": "2024",
                        "wins": 94,
                        "losses": 68,
                        "games_played": 162,
                        "winning_percentage": 0.58,
                    }
                ]
            }
        )
    ]
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    standings = parse_standings(client.standings(2024))
    mlb.get_standings.assert_called_once()
    args, kwargs = mlb.get_standings.call_args
    assert args[:2] == ("103,104", "2024")
    assert kwargs["sportId"] == 1
    assert standings.iloc[0]["wins"] == 94
    assert standings.iloc[0]["winning_pct"] == pytest.approx(0.58)


@pytest.mark.unit
def test_schedule_non_mapping_dump_returns_empty_dates() -> None:
    mlb = MagicMock()
    mlb.get_schedule.return_value = ["not-a-schedule"]
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    assert client.schedule(season=2024) == {"dates": []}


@pytest.mark.unit
def test_get_without_fetcher_raises() -> None:
    client = MlbStatsClient(mlb=MagicMock(), min_interval=0)
    with pytest.raises(MlbStatsError, match="test fetcher hook"):
        client.get("/api/v1/teams", {"sportId": 1})


@pytest.mark.unit
def test_team_stats_skips_teams_without_id() -> None:
    mlb = MagicMock()
    mlb.get_teams.return_value = [
        _Dump({"name": "No Id"}),
        _Dump({"id": 147, "name": "Yankees"}),
    ]
    mlb.get_team_stats.return_value = {
        "hitting": {
            "season": _Dump(
                {
                    "splits": [
                        {
                            "season": "2024",
                            "team": {"id": 147, "name": "Yankees"},
                            "stat": {"home_runs": 237, "games_played": 162},
                        }
                    ]
                }
            )
        }
    }
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    hitting = parse_team_stats(client.team_stats(2024, "hitting"), "hitting")
    mlb.get_team_stats.assert_called_once()
    assert mlb.get_team_stats.call_args.args[0] == 147
    assert list(hitting["mlb_team_id"]) == [147]


@pytest.mark.unit
def test_team_stats_raises_when_every_team_fails() -> None:
    from mlbstatsapi import MlbHttpError

    mlb = MagicMock()
    mlb.get_teams.return_value = [
        _Dump({"id": 147, "name": "Yankees"}),
        _Dump({"id": 133, "name": "Athletics"}),
    ]
    mlb.get_team_stats.side_effect = MlbHttpError(503, "unavailable", url="/teams/stats")
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    with pytest.raises(MlbStatsError) as excinfo:
        client.team_stats(2024, "hitting")
    assert excinfo.value.status_code == 503
    assert mlb.get_team_stats.call_count == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("exc", "match"),
    [
        ("transport", "transport failed"),
        ("decode", "Invalid JSON"),
        ("generic", "request failed"),
    ],
)
def test_client_maps_remaining_library_errors(exc: str, match: str) -> None:
    from mlbstatsapi import MlbDecodeError, MlbTransportError, TheMlbStatsApiException

    errors = {
        "transport": MlbTransportError("connection reset"),
        "decode": MlbDecodeError("truncated body"),
        "generic": TheMlbStatsApiException("unexpected library failure"),
    }
    mlb = MagicMock()
    mlb.get_teams.side_effect = errors[exc]
    client = MlbStatsClient(mlb=mlb, min_interval=0)
    with pytest.raises(MlbStatsError, match=match):
        client.teams()


@pytest.mark.unit
def test_cli_closes_client_after_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        "raw_dir: raw\nartifacts_uri: ''\nartifacts_dir: artifacts\nmlb_stats: {}\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    fake_client = MagicMock()
    monkeypatch.setattr(pull_mod, "client_from_settings", lambda _settings: fake_client)

    def ok_report(*_args, **_kwargs):
        return ExtractReport(
            as_of_date=AS_OF,
            seasons=[2024],
            endpoints=[EndpointResult(endpoint="teams", ok=True)],
        )

    monkeypatch.setattr(pull_mod, "pull_majors_feeds", ok_report)
    success = CliRunner().invoke(
        pull_mod.app,
        ["--config-path", str(settings_path), "--as-of-date", AS_OF],
    )
    assert success.exit_code == 0
    fake_client.close.assert_called_once()

    fake_client.reset_mock()

    def boom(*_args, **_kwargs):
        raise RuntimeError("statsapi down")

    monkeypatch.setattr(pull_mod, "pull_majors_feeds", boom)
    failed = CliRunner().invoke(
        pull_mod.app,
        ["--config-path", str(settings_path), "--as-of-date", AS_OF],
    )
    assert failed.exit_code == 0
    fake_client.close.assert_called_once()
