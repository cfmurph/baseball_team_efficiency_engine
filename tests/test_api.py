"""Fixture tests for the thin read API over published current/ (#106)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from src.baseball_analytics.config import ArtifactSettings
from src.baseball_analytics.fantasy import FANTASY_SCHEMA_VERSION, card_schema_errors
from src.baseball_analytics.published import redact_secrets
from src.baseball_analytics.storage import FileBackend
from services.api.app import create_app

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "api"
LAKE_CURRENT = FIXTURES / "lake_current"
LAKE_MISSING = FIXTURES / "lake_missing_2026"
PINNED_ENV = {"ARTIFACTS_AS_OF_DATE": "2026-08-23"}
SECRET = "super-secret-sdio-key-not-for-ci"


def _settings(tmp_path: Path, lake: Path | None = None, **overrides) -> ArtifactSettings:
    defaults = dict(
        uri=f"file://{lake}" if lake is not None else None,
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    defaults.update(overrides)
    return ArtifactSettings(**defaults)


def _client(tmp_path: Path, lake: Path | None = None, environ: dict | None = None) -> TestClient:
    env = dict(PINNED_ENV)
    if environ:
        env.update(environ)
    settings = _settings(tmp_path, lake)
    backend = FileBackend(lake) if lake is not None else None
    app = create_app(settings=settings, backend=backend, environ=env)
    return TestClient(app)


def test_health_from_fixture_current(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    body = client.get("/v1/health").json()
    assert body["as_of"] == "2026-08-23"
    assert body["active_season"] == 2026
    assert body["current_season_missing"] is False
    assert body["season_window"] == [2024, 2025, 2026]
    assert body["source"] in {"remote", "local"}
    assert 2026 in body["seasons_present"]


def test_seasons_default_window_is_y_minus_2_through_y(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    body = client.get("/v1/seasons").json()
    assert body["season_window"] == [2024, 2025, 2026]
    assert body["active_season"] == 2026
    assert body["current_season_missing"] is False
    assert 2026 in body["seasons_present"]


def test_cards_schema_1_0_and_stat_line_rules(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    body = client.get("/v1/cards").json()
    assert body["schema_version"] == FANTASY_SCHEMA_VERSION
    assert body["as_of"] == "2026-08-23"
    assert body["current_season_missing"] is False
    cards = body["cards"]
    assert {card["recommendation_type"] for card in cards} == {
        "start",
        "sit",
        "pickup",
        "stream",
    }
    for card in cards:
        assert card_schema_errors(card) == []
        assert card["schema_version"] == "1.0"
        stat = card["share"]["stat_line"]
        assert "vs repl" not in stat
        assert "vs replacement" not in stat
        source = card["edge"]["war_source"]
        assert source in {"bbref", "approx"}
        if source == "approx":
            assert stat.endswith(" edge")
            assert "% conf" not in stat
        else:
            assert " edge · " in stat
            assert stat.endswith("% conf")

    start = next(card for card in cards if card["recommendation_type"] == "start")
    assert start["share"]["stat_line"] == "+3.4 edge · 91% conf"
    sit = next(card for card in cards if card["recommendation_type"] == "sit")
    assert sit["share"]["stat_line"] == "-0.4 edge"


def test_cards_filter_season_and_rec(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    body = client.get("/v1/cards", params={"season": 2026, "rec": "start"}).json()
    assert body["season"] == 2026
    assert body["rec"] == "start"
    assert len(body["cards"]) == 1
    assert body["cards"][0]["player"]["name"] == "Aaron Judge"
    assert body["cards"][0]["share"]["stat_line"] == "+3.4 edge · 91% conf"


def test_stat_line_is_verbatim_not_rewritten(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    fantasy = lake / "current" / "fantasy"
    fantasy.mkdir(parents=True)
    dirty = "+1.0 vs repl"
    record = {
        "schema_version": "1.0",
        "as_of_date": "2026-08-23",
        "season": 2026,
        "card_id": "dirty-1",
        "recommendation_type": "start",
        "player": {"player_id": "x01", "name": "Dirty Line"},
        "edge": {
            "vs_replacement": 1.0,
            "war": 1.0,
            "war_source": "bbref",
            "is_approx": False,
            "confidence": 0.5,
        },
        "reason": "one line",
        "rank": {"among_rec_type": 1},
        "share": {"stat_line": dirty},
    }
    (fantasy / "cards.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (lake / "current" / "metrics").mkdir(parents=True)
    (lake / "current" / "metrics" / "player_season_metrics.csv").write_text(
        "player_id,season,year_id\nx01,2026,2026\n",
        encoding="utf-8",
    )
    client = _client(tmp_path, lake)
    body = client.get("/v1/cards").json()
    assert body["cards"][0]["share"]["stat_line"] == dirty


def test_missing_current_season_does_not_invent_2026(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_MISSING)
    health = client.get("/v1/health").json()
    assert health["as_of"] == "2026-08-23"
    assert health["active_season"] == 2026
    assert health["season_window"] == [2024, 2025, 2026]
    assert health["current_season_missing"] is True
    assert 2026 not in health["seasons_present"]
    assert 2024 in health["seasons_present"]

    seasons = client.get("/v1/seasons").json()
    assert seasons["season_window"] == [2024, 2025, 2026]
    assert 2026 not in seasons["seasons_present"]
    assert seasons["current_season_missing"] is True

    empty = client.get("/v1/cards", params={"season": 2026}).json()
    assert empty["cards"] == []
    assert empty["current_season_missing"] is True
    assert empty["season"] == 2026

    prior = client.get("/v1/cards", params={"season": 2024}).json()
    assert prior["cards"]
    assert all(card["season"] == 2024 for card in prior["cards"])
    assert all(card["season"] != 2026 for card in prior["cards"])


def test_empty_artifacts_do_not_invent_cards_or_2026(tmp_path: Path) -> None:
    client = _client(tmp_path, lake=None, environ=PINNED_ENV)
    health = client.get("/v1/health").json()
    assert health["current_season_missing"] is True
    assert health["season_window"] == [2024, 2025, 2026]
    assert health["source"] == "missing"
    assert 2026 not in health["seasons_present"]
    cards = client.get("/v1/cards", params={"season": 2026}).json()
    assert cards["cards"] == []
    seasons = client.get("/v1/seasons").json()
    assert seasons["seasons_present"] == []
    assert 2026 in seasons["season_window"]
    players = client.get("/v1/players", params={"season": 2026}).json()
    assert players["players"] == []
    assert players["current_season_missing"] is True
    missing = client.get("/v1/players/judgeaa01")
    assert missing.status_code == 404


def test_secret_never_appears_in_responses_or_openapi(tmp_path: Path) -> None:
    env = {**PINNED_ENV, "SPORTSDATAIO_API_KEY": SECRET}
    client = _client(tmp_path, LAKE_CURRENT, environ=env)
    for path in (
        "/v1/health",
        "/v1/cards",
        "/v1/seasons",
        "/v1/players",
        "/v1/players/judgeaa01",
        "/openapi.json",
    ):
        text = client.get(path).text
        assert SECRET not in text
        assert "super-secret" not in text
    assert SECRET not in redact_secrets(f"key={SECRET}", env)
    assert "[SPORTSDATAIO_API_KEY]" in redact_secrets(f"key={SECRET}", env)


def test_cors_allowlist_for_vercel_origin(tmp_path: Path) -> None:
    origin = "https://bench-or-start.vercel.app"
    client = _client(
        tmp_path,
        LAKE_CURRENT,
        environ={**PINNED_ENV, "API_CORS_ORIGINS": origin},
    )
    response = client.get("/v1/health", headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == origin
    denied = client.get("/v1/health", headers={"Origin": "https://evil.example"})
    assert denied.headers.get("access-control-allow-origin") != "https://evil.example"


def test_invalid_rec_is_rejected(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    response = client.get("/v1/cards", params={"rec": "bench"})
    assert response.status_code == 422


def test_openapi_spec_is_checked_in_and_matches_app() -> None:
    spec_path = ROOT / "services" / "api" / "openapi.yaml"
    assert spec_path.is_file()
    committed = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    for path in ("/v1/health", "/v1/cards", "/v1/seasons", "/v1/players", "/v1/players/{id}"):
        assert path in committed["paths"]
        assert "get" in committed["paths"][path]
    health_props = committed["components"]["schemas"]["HealthResponse"]["properties"]
    for field in ("as_of", "active_season", "current_season_missing", "season_window"):
        assert field in health_props

    generated = create_app(environ=PINNED_ENV).openapi()
    assert set(committed["paths"]) <= set(generated["paths"])
    assert SECRET not in json.dumps(generated)
    assert "SPORTSDATAIO_API_KEY" not in json.dumps(generated.get("paths", {}))


def test_local_fallback_without_uri(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    dest = local / "current" / "fantasy"
    dest.mkdir(parents=True)
    src = LAKE_CURRENT / "current" / "fantasy" / "cards.jsonl"
    dest.joinpath("cards.jsonl").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    metrics = local / "current" / "metrics"
    metrics.mkdir(parents=True)
    metrics.joinpath("player_season_metrics.csv").write_text(
        (LAKE_CURRENT / "current" / "metrics" / "player_season_metrics.csv").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    app = create_app(settings=_settings(tmp_path, lake=None), environ=PINNED_ENV)
    client = TestClient(app)
    body = client.get("/v1/cards", params={"rec": "start"}).json()
    assert body["cards"][0]["share"]["stat_line"] == "+3.4 edge · 91% conf"
    health = client.get("/v1/health").json()
    assert health["source"] == "local"
    assert health["current_season_missing"] is False


def _assert_honesty(body: dict, *, current_missing: bool) -> None:
    for field in (
        "as_of",
        "active_season",
        "current_season_missing",
        "season_window",
        "source",
        "seasons_present",
        "current_season_missing_reason",
    ):
        assert field in body
    assert body["season_window"] == [2024, 2025, 2026]
    assert body["active_season"] == 2026
    assert body["current_season_missing"] is current_missing


def test_players_directory_from_fixture_current(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    body = client.get("/v1/players").json()
    _assert_honesty(body, current_missing=False)
    assert body["season"] is None
    ids = {player["player_id"] for player in body["players"]}
    assert ids == {"judgeaa01", "solerjo01", "steersp01", "suarera02"}
    judge = next(player for player in body["players"] if player["player_id"] == "judgeaa01")
    assert judge["name"] == "Aaron Judge"
    assert [row["season"] for row in judge["seasons"]] == [2026, 2025]
    season_2026 = judge["seasons"][0]
    assert season_2026["pa"] == 500
    assert season_2026["hr"] == 40
    assert season_2026["runs"] == 85
    assert season_2026["doubles"] == 22
    assert season_2026["avg"] == pytest.approx(0.35)
    assert season_2026["war"] == pytest.approx(6.1)
    assert season_2026["putouts"] == 248
    assert season_2026["fpct"] == pytest.approx(0.988)
    assert season_2026["cs"] == 3
    assert season_2026["hbp"] == 8
    assert season_2026["singles"] == 77
    assert season_2026["xbh"] == 63
    assert season_2026["tb"] == 284
    assert season_2026["tc"] == 258
    assert season_2026["fielding"][0]["pos"] == "RF"
    assert season_2026["fielding"][0]["po"] == 248
    assert season_2026["fielding"][0]["tc"] == 258
    suarez = next(player for player in body["players"] if player["player_id"] == "suarera02")
    pitch = suarez["seasons"][0]
    assert pitch["pitching_hits"] == 120
    assert pitch["cg"] == 2
    assert pitch["qs"] == 18
    assert pitch["bf"] == 610
    assert pitch["wpct"] == pytest.approx(0.6)
    soler = next(player for player in body["players"] if player["player_id"] == "solerjo01")
    assert soler["seasons"][0]["fielding"] == []
    assert soler["seasons"][0]["putouts"] is None
    dumped = json.dumps(body)
    assert "vs_replacement" not in dumped
    assert "vs repl" not in dumped
    assert "salary" not in dumped
    assert "40000000" not in dumped
    assert "dfs" not in dumped.lower()
    assert "betting" not in dumped.lower()


def test_players_season_filter_and_detail(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_CURRENT)
    listed = client.get("/v1/players", params={"season": 2026}).json()
    assert listed["season"] == 2026
    assert {player["player_id"] for player in listed["players"]} == {"judgeaa01", "solerjo01"}
    assert all(row["season"] == 2026 for player in listed["players"] for row in player["seasons"])

    detail = client.get("/v1/players/judgeaa01").json()
    _assert_honesty(detail, current_missing=False)
    assert detail["player"]["player_id"] == "judgeaa01"
    assert [row["season"] for row in detail["player"]["seasons"]] == [2026, 2025]
    assert detail["player"]["seasons"][0]["stat_source"] == "sportsdataio"

    missing = client.get("/v1/players/not-a-player")
    assert missing.status_code == 404
    assert "player not found" in missing.json()["detail"]


def test_players_missing_2026_is_honest_empty(tmp_path: Path) -> None:
    client = _client(tmp_path, LAKE_MISSING)
    empty = client.get("/v1/players", params={"season": 2026}).json()
    _assert_honesty(empty, current_missing=True)
    assert empty["players"] == []
    assert 2026 not in empty["seasons_present"]
    assert empty["current_season_missing_reason"]

    prior = client.get("/v1/players").json()
    _assert_honesty(prior, current_missing=True)
    assert {player["player_id"] for player in prior["players"]} == {"judgeaa01"}
    assert prior["players"][0]["seasons"][0]["season"] == 2024
    assert all(row["season"] != 2026 for player in prior["players"] for row in player["seasons"])

    empty_player = client.get("/v1/players/judgeaa01", params={"season": 2026}).json()
    _assert_honesty(empty_player, current_missing=True)
    assert empty_player["player"]["player_id"] == "judgeaa01"
    assert empty_player["player"]["seasons"] == []

    outside = client.get("/v1/players/troutmi01").json()
    assert outside["player"]["player_id"] == "troutmi01"
    assert outside["player"]["seasons"] == []


def test_players_cors_uses_api_cors_origins(tmp_path: Path) -> None:
    origin = "https://bench-or-start.vercel.app"
    client = _client(
        tmp_path,
        LAKE_CURRENT,
        environ={**PINNED_ENV, "API_CORS_ORIGINS": origin},
    )
    response = client.get("/v1/players", headers={"Origin": origin})
    assert response.headers.get("access-control-allow-origin") == origin
    denied = client.get("/v1/players/judgeaa01", headers={"Origin": "https://evil.example"})
    assert denied.headers.get("access-control-allow-origin") != "https://evil.example"
