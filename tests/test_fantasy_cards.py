from __future__ import annotations

import json
from pathlib import Path

from src.baseball_analytics.config import ArtifactSettings
from src.baseball_analytics.fantasy import (
    FANTASY_CARDS_RELPATH,
    VOID_DATED_CARDS_PREFIX,
    map_card_war_source,
    render_cards_jsonl,
    write_fantasy_cards_stub,
)

from fantasy.cards import (
    CARD_LAKE_KEY,
    RETIRED_CARD_NAMES,
    card_feed_keys,
    card_headline,
    card_rank_line,
    card_stat_line,
    card_subtitle,
    is_approx,
    load_share_cards,
    load_stub_cards,
    parse_cards_jsonl,
    present_card,
    recommendation_label,
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


def test_feed_key_is_current_cards_jsonl_only() -> None:
    assert CARD_LAKE_KEY == "current/fantasy/cards.jsonl"
    assert card_feed_keys() == ("current/fantasy/cards.jsonl",)
    for retired in RETIRED_CARD_NAMES:
        assert retired not in CARD_LAKE_KEY


def test_emitter_path_is_jsonl_not_dated_filename() -> None:
    assert FANTASY_CARDS_RELPATH == "fantasy/cards.jsonl"
    assert "as_of" not in FANTASY_CARDS_RELPATH
    assert VOID_DATED_CARDS_PREFIX not in FANTASY_CARDS_RELPATH


def test_empty_stub_is_valid_and_uses_locked_path(tmp_path: Path) -> None:
    dest = write_fantasy_cards_stub(tmp_path, as_of_date="2026-08-23")
    assert dest == tmp_path / "fantasy" / "cards.jsonl"
    assert dest.read_text(encoding="utf-8") == ""
    assert not (tmp_path / "fantasy" / "fantasy_cards_2026-08-23.json").exists()


def test_records_carry_as_of_date_schema_and_bbref_war_source() -> None:
    text = render_cards_jsonl(
        [
            {"player_id": "judgeaa01", "war_source": "real", "war": 10.8},
            {"player_id": "unknown01", "war_source": "approx", "war": 1.2},
        ],
        as_of_date="2026-08-23",
        schema_version="1.0",
    )
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0]["as_of_date"] == "2026-08-23"
    assert rows[0]["schema_version"] == "1.0"
    assert rows[0]["war_source"] == "bbref"
    assert rows[1]["war_source"] == "approx"
    assert {row["war_source"] for row in rows} <= {"bbref", "approx"}


def test_map_war_source_real_to_bbref() -> None:
    assert map_card_war_source("real") == "bbref"
    assert map_card_war_source("bbref") == "bbref"
    assert map_card_war_source("approx") == "approx"
    assert map_card_war_source("mixed") == "approx"


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
    cards, source = load_share_cards(_settings(tmp_path))
    assert source == "local"
    assert cards[0]["player"]["name"] == "Live"


def test_missing_live_feed_uses_stub(tmp_path: Path) -> None:
    cards, source = load_share_cards(_settings(tmp_path))
    assert source == "stub"
    assert len(cards) == 4


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
    cards, source = load_share_cards(settings, backend=backend)
    assert cards[0]["recommendation_type"] == "pickup"
    assert source == "remote"
    assert "current/fantasy/cards.jsonl" in backend.gets
    assert not any("fantasy_cards_" in key for key in backend.gets)
