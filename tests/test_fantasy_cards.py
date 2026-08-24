from __future__ import annotations

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
    _share_stat_line,
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
    PLAYER_ARTIFACTS,
    RUN_CARD_TEMPLATE,
    SOURCE_MISSING,
    card_feed_keys,
    card_headline,
    card_rank_line,
    card_share_filename,
    card_stat_line,
    card_subtitle,
    cards_for_label,
    normalize_stat_line,
    is_approx,
    load_share_cards,
    load_stub_cards,
    parse_card_payload,
    parse_cards_jsonl,
    present_card,
    present_cards,
    recommendation_label,
    resolve_player_artifacts,
    run_card_keys,
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

@pytest.mark.unit
def test_feed_keys_lock_current_fantasy_cards_jsonl() -> None:
    assert CARD_LAKE_KEY == "current/fantasy/cards.jsonl"
    assert card_feed_keys() == ("current/fantasy/cards.jsonl",)
    assert RUN_CARD_TEMPLATE == "runs/{run_id}/fantasy/cards.jsonl"
    assert all(not key.startswith("fantasy_cards_") for key in card_feed_keys())

@pytest.mark.unit
def test_emitter_path_is_jsonl_not_dated_filename() -> None:
    assert FANTASY_CARDS_RELPATH == "fantasy/cards.jsonl"
    assert "as_of" not in FANTASY_CARDS_RELPATH
    assert VOID_DATED_CARDS_PREFIX not in FANTASY_CARDS_RELPATH

@pytest.mark.integration
def test_empty_stub_is_valid_and_uses_locked_path(tmp_path: Path) -> None:
    dest = write_fantasy_cards_stub(tmp_path, as_of_date="2026-08-23")
    assert dest == tmp_path / "fantasy" / "cards.jsonl"
    assert dest.read_text(encoding="utf-8") == ""
    assert not any(tmp_path.joinpath("fantasy").glob("fantasy_cards_*.json"))

@pytest.mark.unit
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

@pytest.mark.unit
def test_map_war_source_real_to_bbref() -> None:
    assert map_card_war_source("real") == "bbref"
    assert map_card_war_source("bbref") == "bbref"
    assert map_card_war_source("approx") == "approx"
    assert map_card_war_source("mixed") == "approx"
    assert map_card_war_source("fangraphs") == "approx"

@pytest.mark.unit
def test_share_stat_line_edge_and_conf_rules() -> None:
    """Emitter face copy: bbref shows conf; approx does not; never 'vs repl'."""
    assert _share_stat_line(1.64, 0.81, is_approx=False) == "+1.6 edge · 81% conf"
    assert _share_stat_line(-0.44, 0.66, is_approx=True) == "-0.4 edge"
    assert "% conf" not in _share_stat_line(2.2, 0.91, is_approx=True)
    assert "vs repl" not in _share_stat_line(3.4, 0.91, is_approx=False)
    # Values already expressed as percent ( > 1 ) are not multiplied by 100.
    assert _share_stat_line(2.0, 81.4, is_approx=False) == "+2.0 edge · 81% conf"
    assert _share_stat_line(0.0, 0.0, is_approx=False) == "+0.0 edge · 0% conf"

@pytest.mark.unit
def test_recommendation_labels_map_sit_to_bench() -> None:
    assert recommendation_label("start") == "START"
    assert recommendation_label("sit") == "BENCH"
    assert recommendation_label("pickup") == "PICK UP"
    assert recommendation_label("stream") == "STREAM"

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.unit
def test_war_source_approx_implies_early_model_even_without_flag() -> None:
    card = {"edge": {"war_source": "approx", "is_approx": False}}
    assert is_approx(card) is True
    assert war_source(card) == "approx"

@pytest.mark.unit
def test_fangraphs_is_not_a_supported_war_source() -> None:
    assert war_source({"edge": {"war_source": "fangraphs"}}) == ""

@pytest.mark.unit
def test_card_schema_errors_enforces_war_source_and_required_fields() -> None:
    valid = rank_fantasy_cards(_player_metrics_frame(), as_of_date="2026-08-23", top_n=1)
    assert valid
    assert card_schema_errors(valid[0]) == []

    missing = card_schema_errors({})
    assert "missing schema_version" in missing
    assert "missing edge" not in missing or "edge must be an object" in missing
    assert "edge must be an object" in missing

    fangraphs = dict(valid[0])
    fangraphs["edge"] = {**valid[0]["edge"], "war_source": "fangraphs", "is_approx": False}
    errors = card_schema_errors(fangraphs)
    assert "edge.war_source must be bbref|approx" in errors
    assert "edge.war_source cannot be fangraphs" in errors

    approx_inconsistent = dict(valid[0])
    approx_inconsistent["edge"] = {**valid[0]["edge"], "war_source": "approx", "is_approx": False}
    assert "is_approx must be true when war_source is approx" in card_schema_errors(approx_inconsistent)

@pytest.mark.unit
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

@pytest.mark.unit
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

@pytest.mark.e2e
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

@pytest.mark.e2e
def test_local_pipeline_jsonl_is_current_fallback(tmp_path: Path) -> None:
    lake = tmp_path / "artifacts" / "fantasy"
    lake.mkdir(parents=True)
    (lake / "cards.jsonl").write_text(
        '{"recommendation_type": "stream", "as_of_date": "2026-08-23", "player": {"name": "Local"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path), environ={})
    assert feed.source == "local"
    assert feed.key == "current/fantasy/cards.jsonl"
    assert feed.cards[0]["player"]["name"] == "Local"

@pytest.mark.e2e
def test_run_cards_jsonl_when_current_missing(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "runs" / "2026-08-23" / "fantasy"
    run.mkdir(parents=True)
    (run / "cards.jsonl").write_text(
        '{"recommendation_type": "stream", "as_of_date": "2026-08-23", "player": {"name": "Run"}}',
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "fantasy_cards_2026-08-23.json").write_text(
        '{"cards": [{"recommendation_type": "sit", "player": {"name": "Dated"}}]}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path), environ={})
    assert feed.source == "local"
    assert feed.key == "runs/2026-08-23/fantasy/cards.jsonl"
    assert feed.cards[0]["player"]["name"] == "Run"

@pytest.mark.e2e
def test_current_cards_win_over_run(tmp_path: Path) -> None:
    current = tmp_path / "artifacts" / "current" / "fantasy"
    run = tmp_path / "artifacts" / "runs" / "2026-08-22" / "fantasy"
    current.mkdir(parents=True)
    run.mkdir(parents=True)
    (current / "cards.jsonl").write_text(
        '{"recommendation_type": "start", "as_of_date": "2026-08-23", "player": {"name": "Current"}}',
        encoding="utf-8",
    )
    (run / "cards.jsonl").write_text(
        '{"recommendation_type": "sit", "as_of_date": "2026-08-22", "player": {"name": "Run"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path), environ={})
    assert feed.cards[0]["player"]["name"] == "Current"
    assert feed.key == "current/fantasy/cards.jsonl"

@pytest.mark.e2e
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

@pytest.mark.integration
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

@pytest.mark.e2e
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

@pytest.mark.e2e
def test_remote_run_cards_when_current_missing(tmp_path: Path) -> None:
    backend = _MemoryBackend(
        {"runs/2026-08-23/fantasy/cards.jsonl": b'{"recommendation_type": "sit"}\n'}
    )
    feed = load_share_cards(
        _settings(tmp_path, uri="s3://bucket/prefix"),
        backend=backend,
        environ={"ARTIFACTS_RUN_DATE": "2026-08-23"},
    )
    assert feed.cards[0]["recommendation_type"] == "sit"
    assert feed.key == "runs/2026-08-23/fantasy/cards.jsonl"
    assert feed.source == "remote"
    assert not any("fantasy_cards_" in key for key in backend.gets)

@pytest.mark.e2e
def test_run_id_env_selects_dated_run(tmp_path: Path) -> None:
    older = tmp_path / "artifacts" / "runs" / "nightly-old" / "fantasy"
    newer = tmp_path / "artifacts" / "runs" / "nightly-new" / "fantasy"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "cards.jsonl").write_text(
        '{"recommendation_type": "start", "as_of_date": "2026-08-01", "player": {"name": "Old"}}',
        encoding="utf-8",
    )
    (newer / "cards.jsonl").write_text(
        '{"recommendation_type": "sit", "as_of_date": "2026-08-22", "player": {"name": "New"}}',
        encoding="utf-8",
    )
    feed = load_share_cards(
        _settings(tmp_path),
        environ={"ARTIFACTS_RUN_ID": "nightly-old"},
    )
    assert feed.cards[0]["player"]["name"] == "Old"
    assert feed.key == "runs/nightly-old/fantasy/cards.jsonl"

@pytest.mark.unit
def test_parse_card_payload_is_jsonl_only() -> None:
    cards = parse_card_payload(
        '{"recommendation_type": "start"}\n{"recommendation_type": "sit"}\n'
    )
    assert [card["recommendation_type"] for card in cards] == ["start", "sit"]
    assert (
        parse_card_payload(
            '[{"recommendation_type": "start"}]',
            filename="fantasy_cards_2026-08-23.json",
        )
        == []
    )
    assert (
        parse_card_payload(
            '{"schema_version": "1.0", "cards": [{"recommendation_type": "pickup"}]}',
            filename="fantasy_cards_2026-08-23.json",
        )
        == []
    )

@pytest.mark.integration
def test_dated_fantasy_cards_json_is_ignored(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "fantasy_cards_2026-08-23.json").write_text(
        '{"cards": [{"recommendation_type": "stream", "player": {"name": "Dated"}}]}',
        encoding="utf-8",
    )
    feed = load_share_cards(_settings(tmp_path), environ={})
    assert feed.source == SOURCE_MISSING
    assert feed.cards == []
    assert load_stub_cards()

@pytest.mark.e2e
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

@pytest.mark.e2e
def test_run_card_keys_from_env_and_local_scan(tmp_path: Path) -> None:
    run = tmp_path / "artifacts" / "runs" / "2026-07-04" / "fantasy"
    run.mkdir(parents=True)
    (run / "cards.jsonl").write_text("{}\n", encoding="utf-8")
    keys = run_card_keys(
        _settings(tmp_path),
        environ={"ARTIFACTS_RUN_ID": "abc", "ARTIFACTS_RUN_DATE": "2026-07-04"},
    )
    assert keys[0] == "runs/abc/fantasy/cards.jsonl"
    assert "runs/2026-07-04/fantasy/cards.jsonl" in keys
    assert all(not key.startswith("fantasy_cards_") for key in keys)

@pytest.mark.integration
def test_rank_and_emit_ranked_cards_jsonl(tmp_path: Path) -> None:
    """Ranked emitter writes locked fantasy/cards.jsonl with share.stat_line rules."""
    cards = rank_fantasy_cards(_player_metrics_frame(), as_of_date="2026-08-23", top_n=2)
    _assert_schema_cards(cards)
    names = {card["player"]["name"] for card in cards}
    assert "Prior Season" not in names

    dest = emit_ranked_fantasy_cards(
        tmp_path,
        as_of_date="2026-08-23",
        player_df=_player_metrics_frame(),
        top_n=2,
    )
    assert dest == tmp_path / "fantasy" / "cards.jsonl"
    written = [json.loads(line) for line in dest.read_text(encoding="utf-8").splitlines()]
    _assert_schema_cards(written)
    dated = tmp_path / "fantasy" / "fantasy_cards_2015.json"
    dated.write_text("{}", encoding="utf-8")
    emit_ranked_fantasy_cards(
        tmp_path,
        as_of_date="2026-08-23",
        player_df=_player_metrics_frame(),
        top_n=1,
    )
    assert not dated.exists()

@pytest.mark.integration
def test_emit_empty_metrics_writes_empty_stub(tmp_path: Path) -> None:
    dest = emit_ranked_fantasy_cards(
        tmp_path,
        as_of_date="2026-08-23",
        player_df=pd.DataFrame(),
    )
    assert dest == tmp_path / "fantasy" / "cards.jsonl"
    assert dest.read_text(encoding="utf-8") == ""

@pytest.mark.integration
def test_player_artifacts_use_shared_resolve(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "player_season_metrics.csv").write_text("player_id\n", encoding="utf-8")
    resolved = resolve_player_artifacts(_settings(tmp_path, uri=None))
    assert set(resolved) == set(PLAYER_ARTIFACTS)
    assert resolved["players"] == local / "player_season_metrics.csv"
    assert resolved["top_value"] is None

@pytest.mark.unit
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

@pytest.mark.unit
def test_share_blurb_is_league_chat_ready() -> None:
    card = load_stub_cards()[0]
    view = present_card(card)
    blurb = share_blurb(view)
    assert blurb.startswith("PICK UP — Spencer Steer")
    assert "+1.6 edge" in blurb
    assert "vs repl" not in blurb
    assert "Quiet week on the wire" in blurb
    assert "as of 2026-08-23" in blurb

@pytest.mark.unit
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

@pytest.mark.unit
def test_tabs_filter_by_recommendation_label() -> None:
    views = present_cards(load_stub_cards())
    assert [v.label for v in cards_for_label(views, "START")] == ["START"]
    assert [v.label for v in cards_for_label(views, "BENCH")] == ["BENCH"]
    assert len(cards_for_label(views, "All")) == 4

@pytest.mark.unit
def test_share_card_png_and_filename() -> None:
    view = present_card(load_stub_cards()[0])
    png = render_share_card_png(view)
    assert png.startswith(b"\x89PNG")
    assert len(png) > 1000
    assert card_share_filename(view) == "benchorstart-spencer-steer-pickup.png"

@pytest.mark.unit
def test_stub_and_sample_share_stat_line_never_says_vs_repl() -> None:
    for card in load_stub_cards():
        share = card.get("share") or {}
        stat = str(share.get("stat_line") or "").strip()
        assert "vs repl" not in stat.lower()
        if stat:
            assert "edge" in stat
        assert "vs repl" not in card_stat_line(card).lower()

@pytest.mark.unit
def test_rank_fantasy_cards_picks_latest_season_including_2026() -> None:
    frame = _player_metrics_frame()
    current = frame.loc[frame["player_id"] == "judgeaa01"].copy()
    current["season"] = 2026
    current["player_war"] = 8.1
    current["war"] = 8.1
    current["vs_replacement"] = 8.1
    combined = pd.concat([frame, current], ignore_index=True)
    cards = rank_fantasy_cards(combined, as_of_date="2026-08-23", top_n=2)
    assert cards
    assert {card["season"] for card in cards} == {2026}
    start = next(card for card in cards if card["recommendation_type"] == "start")
    assert start["player"]["player_id"] == "judgeaa01"
    assert start["edge"]["vs_replacement"] == pytest.approx(8.1)


@pytest.mark.unit
def test_share_stat_line_vs_repl_is_normalized_on_face_not_schema() -> None:
    dirty = "+3.4 vs replacement · 91% conf"
    dirty_reason = "Aaron Judge is +3.4 vs replacement — lock this OF in."
    card = {
        "recommendation_type": "start",
        "player": {"name": "Aaron Judge", "position": "OF", "team": "NYY"},
        "edge": {"vs_replacement": 3.4, "war_source": "bbref", "is_approx": False},
        "share": {"stat_line": dirty},
        "reason": dirty_reason,
        "as_of_date": "2026-08-23",
    }
    assert card["share"]["stat_line"] == dirty
    assert card["reason"] == dirty_reason
    view = present_card(card)
    assert view.stat_line == "+3.4 edge · 91% conf"
    assert view.reason == "Aaron Judge is +3.4 edge — lock this OF in."
    assert "vs repl" not in view.stat_line.lower()
    assert "vs repl" not in view.reason.lower()
    blurb = share_blurb(view)
    html = share_card_html(view)
    assert "vs repl" not in blurb.lower()
    assert "vs replacement" not in html.lower()
    assert "Aaron Judge is +3.4 edge" in blurb
    assert "Aaron Judge is +3.4 edge" in html
    assert card["share"]["stat_line"] == dirty
    assert card["reason"] == dirty_reason
    assert normalize_stat_line("vs repl") == "edge"
    assert normalize_stat_line("vs replacement") == "edge"
    assert normalize_stat_line("vs replx") == ""
    leftover = view._replace(reason=dirty_reason)
    assert "vs repl" not in share_blurb(leftover).lower()
    assert "vs replacement" not in share_card_html(leftover).lower()
