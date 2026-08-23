from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from src.baseball_analytics.config import ArtifactSettings
from src.baseball_analytics.fantasy import (
    FANTASY_CARDS_RELPATH,
    FANTASY_SCHEMA_VERSION,
    RECOMMENDATION_TYPES,
    VOID_DATED_CARDS_PREFIX,
    card_schema_errors,
    emit_ranked_fantasy_cards,
    map_card_war_source,
    rank_fantasy_cards,
    render_cards_jsonl,
    write_fantasy_cards_stub,
)
from src.baseball_analytics.storage import upload_artifacts

from fantasy.card_image import render_share_card_png
from fantasy.cards import (
    CARD_LAKE_KEY,
    OPTIONAL_CARD_KEY,
    PLAYER_ARTIFACTS,
    SOURCE_MISSING,
    card_feed_keys,
    card_headline,
    card_rank_line,
    card_share_filename,
    card_stat_line,
    card_subtitle,
    cards_for_label,
    normalize_stat_line,
    dated_card_keys,
    is_approx,
    load_share_cards,
    load_stub_cards,
    parse_card_payload,
    parse_cards_jsonl,
    present_card,
    present_cards,
    recommendation_label,
    resolve_player_artifacts,
    share_blurb,
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


def test_emitter_path_is_jsonl_not_dated_filename() -> None:
    assert FANTASY_CARDS_RELPATH == "fantasy/cards.jsonl"
    assert "as_of" not in FANTASY_CARDS_RELPATH
    assert VOID_DATED_CARDS_PREFIX not in FANTASY_CARDS_RELPATH


def test_empty_stub_is_valid_and_uses_locked_path(tmp_path: Path) -> None:
    dest = write_fantasy_cards_stub(tmp_path, as_of_date="2026-08-23")
    assert dest == tmp_path / "fantasy" / "cards.jsonl"
    assert dest.read_text(encoding="utf-8") == ""
    assert not any(tmp_path.joinpath("fantasy").glob("fantasy_cards_*.json"))


def test_records_carry_as_of_date_schema_and_edge_war_source() -> None:
    text = render_cards_jsonl(
        [
            {"player_id": "judgeaa01", "war_source": "real", "war": 10.8},
            {"player_id": "unknown01", "war_source": "approx", "war": 1.2},
            {"player_id": "fg01", "war_source": "fangraphs", "war": 5.0},
        ],
        as_of_date="2026-08-23",
        schema_version="1.0",
    )
    rows = [json.loads(line) for line in text.splitlines()]
    assert rows[0]["as_of_date"] == "2026-08-23"
    assert rows[0]["schema_version"] == "1.0"
    assert rows[0]["edge"]["war_source"] == "bbref"
    assert rows[0]["edge"]["is_approx"] is False
    assert rows[1]["edge"]["war_source"] == "approx"
    assert rows[1]["edge"]["is_approx"] is True
    assert rows[2]["edge"]["war_source"] == "approx"
    sources = {row["edge"]["war_source"] for row in rows}
    assert sources <= {"bbref", "approx"}
    assert "fangraphs" not in sources


def test_map_war_source_real_to_bbref() -> None:
    assert map_card_war_source("real") == "bbref"
    assert map_card_war_source("bbref") == "bbref"
    assert map_card_war_source("approx") == "approx"
    assert map_card_war_source("mixed") == "approx"
    assert map_card_war_source("fangraphs") == "approx"


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
    assert card_headline(card) == "Spencer Steer"
    assert card_headline(card) != recommendation_label("pickup")
    assert card_subtitle(card) == "Spencer Steer · 1B · CIN"
    assert card_stat_line(card) == "+1.6 edge · 81% conf"
    assert "vs repl" not in card_stat_line(card)
    assert card_rank_line(card) == "#1 pickup tonight"
    assert card["edge"]["vs_replacement"] == 1.6

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
    assert card_stat_line(approx) == "-0.4 edge"
    assert "vs repl" not in share_card_html(present_card(approx))
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
    assert "Jorge Soler" in html
    assert html.count("BENCH") == 1
    assert "vs repl" not in html
    pickup = next(card for card in cards if card["recommendation_type"] == "pickup")
    pickup_html = share_card_html(present_card(pickup))
    assert "Spencer Steer" in pickup_html
    assert pickup_html.count("PICK UP") == 1


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


def _player_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_id": "judgeaa01",
                "player_name": "Aaron Judge",
                "team_id": "NYA",
                "position": "OF",
                "player_type": "batter",
                "season": 2015,
                "player_war": 9.4,
                "war": 9.4,
                "vs_replacement": 9.4,
                "surplus_value": 40_000_000,
                "pitching_war": 0.0,
                "war_source": "real",
            },
            {
                "player_id": "troutmi01",
                "player_name": "Mike Trout",
                "team_id": "LAA",
                "position": "OF",
                "player_type": "batter",
                "season": 2015,
                "player_war": 9.0,
                "war": 9.0,
                "vs_replacement": 9.0,
                "surplus_value": 55_000_000,
                "pitching_war": 0.0,
                "war_source": "bbref",
            },
            {
                "player_id": "degroja01",
                "player_name": "Jacob deGrom",
                "team_id": "NYN",
                "position": "P",
                "player_type": "pitcher",
                "season": 2015,
                "player_war": 6.4,
                "war": 6.4,
                "vs_replacement": 6.4,
                "surplus_value": 48_000_000,
                "pitching_war": 6.4,
                "war_source": "real",
            },
            {
                "player_id": "kershcl01",
                "player_name": "Clayton Kershaw",
                "team_id": "LAN",
                "position": "P",
                "player_type": "pitcher",
                "season": 2015,
                "player_war": 5.8,
                "war": 5.8,
                "vs_replacement": 5.8,
                "surplus_value": 12_000_000,
                "pitching_war": 5.8,
                "war_source": "approx",
            },
            {
                "player_id": "steersp01",
                "player_name": "Spencer Steer",
                "team_id": "CIN",
                "position": "1B",
                "player_type": "batter",
                "season": 2015,
                "player_war": 2.4,
                "war": 2.4,
                "vs_replacement": 2.4,
                "surplus_value": 22_000_000,
                "pitching_war": 0.0,
                "war_source": "bbref",
            },
            {
                "player_id": "solerjo01",
                "player_name": "Jorge Soler",
                "team_id": "KCA",
                "position": "OF",
                "player_type": "batter",
                "season": 2015,
                "player_war": -0.6,
                "war": -0.6,
                "vs_replacement": -0.6,
                "surplus_value": -8_000_000,
                "pitching_war": 0.0,
                "war_source": "approx",
            },
            {
                "player_id": "oldbat01",
                "player_name": "Prior Season",
                "team_id": "BOS",
                "position": "OF",
                "player_type": "batter",
                "season": 2014,
                "player_war": 12.0,
                "war": 12.0,
                "vs_replacement": 12.0,
                "surplus_value": 99_000_000,
                "pitching_war": 0.0,
                "war_source": "real",
            },
        ]
    )


def _assert_schema_cards(cards: list[dict]) -> None:
    assert cards
    types = {card["recommendation_type"] for card in cards}
    assert types == set(RECOMMENDATION_TYPES)
    for card in cards:
        assert card_schema_errors(card) == []
        assert card["schema_version"] == FANTASY_SCHEMA_VERSION
        source = card["edge"]["war_source"]
        assert source in {"bbref", "approx"}
        assert source != "fangraphs"
        assert card["edge"]["is_approx"] is (source == "approx")
        assert "\n" not in card["reason"]
        assert "vs replacement" in card["reason"]
        stat_line = str((card.get("share") or {}).get("stat_line") or "")
        assert stat_line
        assert "vs repl" not in stat_line
        assert "vs replacement" not in stat_line
        if source == "approx":
            assert stat_line.endswith(" edge")
            assert "% conf" not in stat_line
        else:
            assert " edge · " in stat_line
            assert stat_line.endswith("% conf")


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
    assert feed.source == "remote"
    assert feed.key == "current/fantasy/cards.jsonl"
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


def test_empty_share_headline_falls_back_to_player_name_not_badge() -> None:
    card = {
        "recommendation_type": "pickup",
        "player": {"name": "Spencer Steer"},
        "share": {"headline": "   "},
    }
    assert card_headline(card) == "Spencer Steer"
    nameless = {"recommendation_type": "pickup", "player": {}, "share": {}}
    assert card_headline(nameless) == ""
    assert recommendation_label("pickup") == "PICK UP"


def test_share_blurb_is_league_chat_ready() -> None:
    card = load_stub_cards()[0]
    view = present_card(card)
    blurb = share_blurb(view)
    assert blurb.startswith("PICK UP — Spencer Steer")
    assert "+1.6 edge" in blurb
    assert "vs repl" not in blurb
    assert "Quiet week on the wire" in blurb
    assert "as of 2026-08-23" in blurb


def test_share_blurb_keeps_custom_headline_and_player() -> None:
    view = present_card(
        {
            "recommendation_type": "stream",
            "player": {"name": "Ranger Suárez", "position": "SP", "team": "PHI"},
            "share": {"headline": "Stream this arm"},
            "reason": "Matchup.",
            "as_of_date": "2026-08-23",
        }
    )
    blurb = share_blurb(view)
    assert "STREAM — Ranger Suárez · SP · PHI" in blurb
    assert "Stream this arm" in blurb


def test_tabs_filter_by_recommendation_label() -> None:
    views = present_cards(load_stub_cards())
    assert [v.label for v in cards_for_label(views, "START")] == ["START"]
    assert [v.label for v in cards_for_label(views, "BENCH")] == ["BENCH"]
    assert len(cards_for_label(views, "All")) == 4


def test_share_card_png_and_filename() -> None:
    view = present_card(load_stub_cards()[0])
    png = render_share_card_png(view)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000
    assert card_share_filename(view) == "benchorstart-spencer-steer-pickup.png"


def test_stub_and_sample_share_stat_line_never_says_vs_repl() -> None:
    for card in load_stub_cards():
        share = card.get("share") or {}
        stat = str(share.get("stat_line") or "").strip()
        assert "vs repl" not in stat.lower()
        if stat:
            assert "edge" in stat
        assert "vs repl" not in card_stat_line(card).lower()


def test_share_stat_line_vs_repl_is_normalized_on_face_not_schema() -> None:
    dirty = "+3.4 vs replacement · 91% conf"
    card = {
        "recommendation_type": "start",
        "player": {"name": "Aaron Judge", "position": "OF", "team": "NYY"},
        "edge": {"vs_replacement": 3.4, "war_source": "bbref", "is_approx": False},
        "share": {"stat_line": dirty},
        "reason": "Lock him in.",
        "as_of_date": "2026-08-23",
    }
    assert card["share"]["stat_line"] == dirty
    view = present_card(card)
    assert view.stat_line == "+3.4 edge · 91% conf"
    assert "vs repl" not in view.stat_line.lower()
    blurb = share_blurb(view)
    html = share_card_html(view)
    assert "vs repl" not in blurb.lower()
    assert "vs replacement" not in html.lower()
    assert "edge" in blurb
    assert card["share"]["stat_line"] == dirty
    assert normalize_stat_line("vs repl") == "edge"
    assert normalize_stat_line("vs replacement") == "edge"
    assert normalize_stat_line("vs replx") == ""
