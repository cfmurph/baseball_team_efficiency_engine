"""Unit tests for published current/ snapshot honesty and card filters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baseball_analytics.config import ArtifactSettings
from src.baseball_analytics.published import (
    SECRET_ENV_KEYS,
    card_season,
    filter_cards,
    load_json_artifact,
    load_player_season_rows,
    load_published_cards,
    published_snapshot,
    redact_secrets,
)

pytestmark = pytest.mark.unit

PINNED_ENV = {"ARTIFACTS_AS_OF_DATE": "2026-08-23"}

VALID_CARD = {
    "schema_version": "1.0",
    "as_of_date": "2026-08-23",
    "season": 2024,
    "card_id": "2024-start-judge",
    "recommendation_type": "start",
    "player": {"player_id": "judgeaa01", "name": "Aaron Judge"},
    "edge": {
        "vs_replacement": 10.8,
        "war": 10.8,
        "war_source": "bbref",
        "is_approx": False,
        "confidence": 0.9,
    },
    "reason": "one line",
    "rank": {"among_rec_type": 1},
    "share": {"stat_line": "+10.8 edge · 90% conf"},
}


def _settings(tmp_path: Path) -> ArtifactSettings:
    return ArtifactSettings(
        uri=None,
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )


def _write_metrics(local: Path, csv_text: str, manifest: dict) -> None:
    metrics = local / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "player_season_metrics.csv").write_text(csv_text, encoding="utf-8")
    (metrics / "metrics_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


def test_snapshot_does_not_union_manifest_claimed_seasons(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    _write_metrics(
        local,
        "player_id,season,year_id\njudgeaa01,2024,2024\n",
        {
            "as_of_date": "2026-08-23",
            "active_season": 2026,
            "season_window": [2024, 2025, 2026],
            "seasons_present": [2024, 2025, 2026],
            "current_season_missing": False,
            "current_season_missing_reason": None,
        },
    )
    snap = published_snapshot(_settings(tmp_path), environ=PINNED_ENV)
    assert snap["seasons_present"] == [2024]
    assert 2026 not in snap["seasons_present"]
    assert snap["active_season"] == 2026
    assert snap["current_season_missing"] is True
    assert snap["current_season_missing_reason"] == "active_season_absent"
    assert snap["season_window"] == [2024, 2025, 2026]


def test_snapshot_keeps_manifest_reason_when_active_year_is_absent(
    tmp_path: Path,
) -> None:
    local = tmp_path / "artifacts"
    _write_metrics(
        local,
        "player_id,season\njudgeaa01,2024\n",
        {
            "as_of_date": "2026-08-23",
            "active_season": 2026,
            "season_window": [2024, 2025, 2026],
            "seasons_present": [2024, 2026],
            "current_season_missing": True,
            "current_season_missing_reason": "sdio_unavailable",
        },
    )
    snap = published_snapshot(_settings(tmp_path), environ=PINNED_ENV)
    assert snap["seasons_present"] == [2024]
    assert snap["current_season_missing"] is True
    assert snap["current_season_missing_reason"] == "sdio_unavailable"


def test_snapshot_empty_metrics_do_not_invent_2026(tmp_path: Path) -> None:
    snap = published_snapshot(_settings(tmp_path), environ=PINNED_ENV)
    assert snap["source"] == "missing"
    assert snap["seasons_present"] == []
    assert snap["cards"] == []
    assert snap["player_seasons"] == []
    assert snap["current_season_missing"] is True
    assert snap["current_season_missing_reason"] == "metrics_empty"
    assert 2026 in snap["season_window"]
    assert 2026 not in snap["seasons_present"]


def test_load_json_artifact_rejects_invalid_and_non_object(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    metrics = local / "metrics"
    metrics.mkdir(parents=True)
    (metrics / "metrics_manifest.json").write_text("{not-json", encoding="utf-8")
    settings = _settings(tmp_path)
    assert load_json_artifact("metrics_manifest.json", settings, environ=PINNED_ENV) is None

    (metrics / "metrics_manifest.json").write_text("[1, 2]", encoding="utf-8")
    assert load_json_artifact("metrics_manifest.json", settings, environ=PINNED_ENV) is None

    (metrics / "metrics_manifest.json").write_text('{"as_of_date": "2026-08-23"}', encoding="utf-8")
    payload = load_json_artifact("metrics_manifest.json", settings, environ=PINNED_ENV)
    assert payload == {"as_of_date": "2026-08-23"}


def test_filter_cards_rejects_unknown_rec_and_does_not_invent() -> None:
    cards = [
        {**VALID_CARD, "season": 2024, "recommendation_type": "start"},
        {**VALID_CARD, "season": "not-a-year", "card_id": "bad-year", "recommendation_type": "sit"},
        {**VALID_CARD, "season": 2026, "card_id": "sit-2026", "recommendation_type": "SIT"},
    ]
    with pytest.raises(ValueError, match="start|sit|pickup|stream"):
        filter_cards(cards, rec="bench")
    assert filter_cards(cards, season=2026) == [cards[2]]
    assert filter_cards(cards, rec="sit") == [cards[1], cards[2]]
    assert filter_cards(cards, season=2025) == []


def test_card_season_is_strict_int() -> None:
    assert card_season({"season": 2026}) == 2026
    assert card_season({"season": "2024"}) == 2024
    assert card_season({"season": "2026.0"}) is None
    assert card_season({"season": ""}) is None
    assert card_season({}) is None


def test_redact_secrets_covers_every_vendor_key_and_ignores_blanks() -> None:
    env = {
        "SPORTSDATAIO_API_KEY": "sdio-secret",
        "SPORTRADAR_API_KEY": "sr-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
        "AWS_SESSION_TOKEN": "session-token",
        "OTHER": "keep-me",
    }
    dumped = "sdio-secret sr-secret aws-secret AKIAEXAMPLE session-token keep-me"
    redacted = redact_secrets(dumped, env)
    assert "sdio-secret" not in redacted
    assert "sr-secret" not in redacted
    assert "aws-secret" not in redacted
    assert "AKIAEXAMPLE" not in redacted
    assert "session-token" not in redacted
    assert "keep-me" in redacted
    for key in SECRET_ENV_KEYS:
        assert f"[{key}]" in redacted
    assert redact_secrets("plain", {"SPORTSDATAIO_API_KEY": "  "}) == "plain"
    assert redact_secrets("plain", None) == "plain"


def test_load_published_cards_drops_invalid_schema_rows(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    fantasy = local / "fantasy"
    fantasy.mkdir(parents=True)
    (fantasy / "cards.jsonl").write_text(
        json.dumps(VALID_CARD)
        + "\n"
        + json.dumps({"schema_version": "1.0", "reason": "incomplete"})
        + "\n{not-json}\n",
        encoding="utf-8",
    )
    cards, source = load_published_cards(_settings(tmp_path), environ=PINNED_ENV)
    assert source == "local"
    assert [card["card_id"] for card in cards] == ["2024-start-judge"]


def test_load_player_season_rows_missing_file_is_empty(tmp_path: Path) -> None:
    assert load_player_season_rows(_settings(tmp_path), environ=PINNED_ENV) == []
    cards, source = load_published_cards(_settings(tmp_path), environ=PINNED_ENV)
    assert cards == []
    assert source == "missing"
