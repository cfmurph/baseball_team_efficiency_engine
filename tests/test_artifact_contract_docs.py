"""Locked contract text lives in ADRs and the operator QA guide."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baseball_analytics.fantasy import FANTASY_CARDS_RELPATH
from src.baseball_analytics.storage import REQUIRED_MANIFEST_FIELDS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def test_adr_layout_uses_runs_current_and_cards_jsonl() -> None:
    adr = (ROOT / "docs/adr/0001-shared-artifact-contract.md").read_text(encoding="utf-8")
    assert "runs/{run_id}/" in adr
    assert "current/" in adr
    assert FANTASY_CARDS_RELPATH in adr
    assert "edge.war_source" in adr
    assert "schema_version" in adr
    assert "empty stub" in adr.lower()
    assert "deprecated" in adr.lower()
    assert "dropped next release" in adr
    assert "{league}/{level}/latest/" in adr
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in adr
    assert "remote" in adr and "local" in adr and "missing" in adr


def test_sot_map_names_br_stats_api_and_lahman() -> None:
    sot = (ROOT / "docs/adr/0002-source-of-truth-map.md").read_text(encoding="utf-8")
    assert "Baseball-Reference rWAR" in sot
    assert "MLB Stats API" in sot
    assert "Lahman" in sot
    assert "No dual-write WAR" in sot
    ingest = (ROOT / "docs/adr/0003-mlb-stats-api-ingest.md").read_text(encoding="utf-8")
    assert "#108" in ingest
    assert "{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/" in ingest
    assert "data/raw/mlb_stats/" in ingest
    assert "pull_war" in ingest
    assert "Lahman" in ingest


def test_phase0_schema_v01_is_locked() -> None:
    root = ROOT / "docs/architecture"
    md = (root / "phase0-schema-v0.1.md").read_text(encoding="utf-8")
    raw = json.loads((root / "phase0-schema-v0.1.json").read_text(encoding="utf-8"))
    assert "Status: **LOCKED by Cole 2026-08-23**" in md
    assert "raw/sportsdataio/{endpoint}/{as_of_date}/" in md
    assert "external_id_alias" in md
    assert "player_game_stat" in md
    assert "SPORTSDATAIO_API_KEY" in md
    assert "fantasy_*_stat" in md
    assert "Clarifying addendum (#131" in md
    assert "season_year ∈ [Y-2, Y]" in md
    assert "Active season in `current/`" in md
    assert "current_season_missing" in md
    assert raw["status"] == "LOCKED"
    assert raw["locked_by"] == "Cole"
    assert raw["locked_on"] == "2026-08-23"
    assert raw["schema_version"] == "0.1"
    assert raw["primary_live_ingest"] == "sportsdataio"
    assert raw["lake"]["raw_prefix"] == "raw/sportsdataio/{endpoint}/{as_of_date}/"
    assert raw["alias_systems"] == ["sportsdataio", "mlb", "bbref", "fangraphs", "lahman"]
    assert raw["spine"]["player_game_stat"]["pk"] == ["player_id", "game_id"]
    assert "source" in raw["provenance"]
    assert "is_approx" in raw["provenance"]
    assert raw["clarifications"]["issue"] == 131
    assert raw["clarifications"]["default_season_window"] == "[Y-2, Y]"
    assert raw["clarifications"]["live_path_for_in_season"] == "sportsdataio"
    assert raw["soft_fail"]["must_not_pretend_current_season"] is True


def test_qa_guide_documents_file_uri_how_to_verify() -> None:
    guide = (ROOT / "docs/shared_artifacts.md").read_text(encoding="utf-8")
    assert "ARTIFACTS_URI=file://" in guide
    assert "file:///tmp/btee-qa" in guide
    assert "current/fantasy/cards.jsonl" in guide
    assert "deprecated" in guide.lower()
    assert "dropped next release" in guide
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in guide
