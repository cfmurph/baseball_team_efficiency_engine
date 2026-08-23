from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.baseball_analytics.config import ArtifactSettings

from fantasy.cards import (
    CARD_LAKE_KEY,
    OPTIONAL_CARD_KEY,
    PLAYER_ARTIFACTS,
    SOURCE_MISSING,
    card_feed_keys,
    card_headline,
    card_rank_line,
    card_stat_line,
    card_subtitle,
    dated_card_keys,
    is_approx,
    load_share_cards,
    load_stub_cards,
    parse_card_payload,
    parse_cards_jsonl,
    present_card,
    recommendation_label,
    resolve_player_artifacts,
    share_card_html,
    war_source,
)
from fantasy.copy import EARLY_MODEL_BADGE, PROMPT_LINE, PRODUCT_NAME


def _settings(tmp_path: Path, **overrides) -> ArtifactSettings:
    defaults = dict(
        uri=None,
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    defaults.update(overrides)
    return ArtifactSettings(**defaults)


def test_feed_keys_lock_current_fantasy_cards_jsonl() -> None:
    assert CARD_LAKE_KEY == "current/fantasy/cards.jsonl"
    assert card_feed_keys()[0] == "current/fantasy/cards.jsonl"
    assert all(not key.startswith("fantasy_cards_") for key in card_feed_keys())


def test_recommendation_labels_map_sit_to_bench() -> None:
    assert recommendation_label("start") == "START"
    assert recommendation_label("sit") == "BENCH"
    assert recommendation_label("pickup") == "PICK UP"
    assert recommendation_label("stream") == "STREAM"


def test_share_fallbacks_and_approx_hides_confidence() -> None:
    card = {
        "recommendation_type": "pickup",
        "as_of_date": "2026-08-23",
        "player": {"name": "Spencer Steer", "position": "1B", "team": "CIN"},
        "edge": {
            "vs_replacement": 1.6,
            "war_source": "bbref",
            "is_approx": False,
            "confidence": 0.81,
        },
        "reason": "Grab him.",
        "rank": {"among_rec_type": 1},
        "share": {},
    }
    assert card_headline(card) == "PICK UP"
    assert card_subtitle(card) == "Spencer Steer · 1B · CIN"
    assert card_stat_line(card) == "+1.6 vs repl · 81% conf"
    assert card_rank_line(card) == "#1 pickup tonight"

    approx = {
        **card,
        "recommendation_type": "sit",
        "edge": {
            "vs_replacement": -0.4,
            "war_source": "approx",
            "is_approx": True,
            "confidence": 0.66,
        },
        "share": {},
    }
    assert is_approx(approx) is True
    assert war_source(approx) == "approx"
    assert card_stat_line(approx) == "-0.4 vs repl"
    view = present_card(approx)
    assert view.label == "BENCH"
    assert view.early_model is True
    assert view.prompt == PROMPT_LINE


def test_share_overrides_win_when_present() -> None:
    card = {
        "recommendation_type": "stream",
        "share": {
            "headline": "Stream this arm",
            "subtitle": "Custom sub",
            "stat_line": "custom stats",
        },
        "edge": {"war_source": "bbref", "is_approx": False},
    }
    assert card_headline(card) == "Stream this arm"
    assert card_subtitle(card) == "Custom sub"
    assert card_stat_line(card) == "custom stats"


def test_war_source_approx_implies_early_model_even_without_flag() -> None:
    card = {"edge": {"war_source": "approx", "is_approx": False}}
    assert is_approx(card) is True
    assert war_source(card) == "approx"


def test_fangraphs_is_not_a_supported_war_source() -> None:
    assert war_source({"edge": {"war_source": "fangraphs"}}) == ""


def test_stub_cards_cover_all_decisions_and_bbref_or_approx() -> None:
    cards = load_stub_cards()
    types = {card["recommendation_type"] for card in cards}
    assert types == {"pickup", "stream", "start", "sit"}
    assert len(cards) == 4
    for card in cards:
        source = card["edge"]["war_source"]
        assert source in {"bbref", "approx"}
        if source == "approx":
            assert card["edge"]["is_approx"] is True
    sit = next(card for card in cards if card["recommendation_type"] == "sit")
    html = share_card_html(present_card(sit))
    assert PRODUCT_NAME in html
    assert PROMPT_LINE in html
    assert "BENCH" in html
    assert EARLY_MODEL_BADGE in html
    assert "as of 2026-08-23" in html


def test_parse_cards_jsonl_skips_bad_lines() -> None:
    text = "\n".join(
        [
            '{"recommendation_type": "start"}',
            "not-json",
            "",
            '{"recommendation_type": "sit"}',
        ]
    )
    assert [card["recommendation_type"] for card in parse_cards_jsonl(text)] == [
        "start",
        "sit",
    ]


def test_live_cards_jsonl_beats_stub(tmp_path: Path) -> None:
    lake = tmp_path / "artifacts" / "current" / "fantasy"
    lake.mkdir(parents=True)
    (lake / "cards.jsonl").write_text(
        '{"recommendation_type": "start", "player": {"name": "Live"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path))
    assert feed.source == "local"
    assert feed.key == "current/fantasy/cards.jsonl"
    assert feed.cards[0]["player"]["name"] == "Live"


def test_optional_fantasy_cards_jsonl_when_current_missing(tmp_path: Path) -> None:
    lake = tmp_path / "artifacts" / "fantasy"
    lake.mkdir(parents=True)
    (lake / "cards.jsonl").write_text(
        '{"recommendation_type": "stream", "player": {"name": "Optional"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path))
    assert feed.source == "local"
    assert feed.key == "fantasy/cards.jsonl"
    assert feed.cards[0]["player"]["name"] == "Optional"


def test_current_cards_win_over_optional(tmp_path: Path) -> None:
    current = tmp_path / "artifacts" / "current" / "fantasy"
    optional = tmp_path / "artifacts" / "fantasy"
    current.mkdir(parents=True)
    optional.mkdir(parents=True)
    (current / "cards.jsonl").write_text(
        '{"recommendation_type": "start", "player": {"name": "Current"}}',
        encoding="utf-8",
    )
    (optional / "cards.jsonl").write_text(
        '{"recommendation_type": "sit", "player": {"name": "Optional"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path))
    assert feed.cards[0]["player"]["name"] == "Current"
    assert feed.key == "current/fantasy/cards.jsonl"


def test_missing_live_feed_is_empty_not_error(tmp_path: Path) -> None:
    feed = load_share_cards(_settings(tmp_path))
    assert feed.source == SOURCE_MISSING
    assert feed.cards == []
    assert load_stub_cards()  # samples stay available for the empty-state UI


class _MemoryBackend:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.gets: list[str] = []

    def get(self, relative_key: str) -> bytes | None:
        self.gets.append(relative_key)
        return self.objects.get(relative_key)


def test_resolve_prefers_current_fantasy_cards_jsonl(tmp_path: Path) -> None:
    backend = _MemoryBackend(
        {
            "current/fantasy/cards.jsonl": b'{"recommendation_type": "pickup"}\n',
            "mlb/mlb/latest/fantasy/fantasy_cards_2026-08-23.json": b"retired",
        }
    )
    settings = _settings(tmp_path, uri="s3://bucket/prefix")
    feed = load_share_cards(settings, backend=backend)
    assert feed.cards[0]["recommendation_type"] == "pickup"
    assert feed.source.startswith("shared")
    assert "current/fantasy/cards.jsonl" in backend.gets
    assert not any("fantasy_cards_" in key for key in backend.gets)


def test_remote_optional_fantasy_cards_jsonl(tmp_path: Path) -> None:
    backend = _MemoryBackend(
        {"fantasy/cards.jsonl": b'{"recommendation_type": "sit"}\n'}
    )
    feed = load_share_cards(_settings(tmp_path, uri="s3://bucket/prefix"), backend=backend)
    assert feed.cards[0]["recommendation_type"] == "sit"
    assert feed.key == "fantasy/cards.jsonl"


def test_parse_fantasy_cards_json_array_and_wrapper() -> None:
    array = parse_card_payload(
        '[{"recommendation_type": "start"}, {"recommendation_type": "sit"}]',
        filename="fantasy_cards_2026-08-23.json",
    )
    assert [row["recommendation_type"] for row in array] == ["start", "sit"]
    wrapped = parse_card_payload(
        '{"schema_version": "1.0", "cards": [{"recommendation_type": "pickup"}]}',
        filename="fantasy_cards_2026-08-23.json",
    )
    assert wrapped[0]["recommendation_type"] == "pickup"


def test_dated_json_used_when_jsonl_missing(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "fantasy_cards_2026-08-23.json").write_text(
        '{"cards": [{"recommendation_type": "stream", "player": {"name": "Dated"}}]}',
        encoding="utf-8",
    )
    feed = load_share_cards(
        _settings(tmp_path),
        now=datetime(2026, 8, 23, tzinfo=timezone.utc),
        environ={},
    )
    assert feed.cards[0]["player"]["name"] == "Dated"
    assert feed.key == "fantasy_cards_2026-08-23.json"


def test_jsonl_wins_over_dated_fantasy_cards(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    current = local / "current" / "fantasy"
    current.mkdir(parents=True)
    (current / "cards.jsonl").write_text(
        '{"recommendation_type": "start", "player": {"name": "Jsonl"}}',
        encoding="utf-8",
    )
    (local / "fantasy_cards_2026-08-23.json").write_text(
        '{"cards": [{"recommendation_type": "sit", "player": {"name": "Dated"}}]}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path), environ={})
    assert feed.cards[0]["player"]["name"] == "Jsonl"
    assert feed.key == "current/fantasy/cards.jsonl"


def test_dated_card_keys_include_configured_date(tmp_path: Path) -> None:
    keys = dated_card_keys(
        _settings(tmp_path),
        environ={"ARTIFACTS_RUN_DATE": "2026-07-04"},
    )
    assert "fantasy_cards_2026-07-04.json" in keys
    assert "current/fantasy/fantasy_cards_2026-07-04.json" in keys


def test_player_artifacts_use_shared_resolve(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "player_season_metrics.csv").write_text("player_id\n", encoding="utf-8")
    resolved = resolve_player_artifacts(_settings(tmp_path, uri=None))
    assert set(resolved) == set(PLAYER_ARTIFACTS)
    assert resolved["players"] == local / "player_season_metrics.csv"
    assert resolved["top_value"] is None
