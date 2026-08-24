"""Fixture tests for the thin read API over published current/ (#106)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.baseball_analytics.config import ArtifactSettings
from src.baseball_analytics.fantasy import FANTASY_SCHEMA_VERSION, card_schema_errors
from src.baseball_analytics.published import redact_secrets
from src.baseball_analytics.storage import FileBackend
from services.api.app import create_app

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


def test_secret_never_appears_in_responses_or_openapi(tmp_path: Path) -> None:
    env = {**PINNED_ENV, "SPORTSDATAIO_API_KEY": SECRET}
    client = _client(tmp_path, LAKE_CURRENT, environ=env)
    for path in ("/v1/health", "/v1/cards", "/v1/seasons", "/openapi.json"):
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
    for path in ("/v1/health", "/v1/cards", "/v1/seasons"):
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
