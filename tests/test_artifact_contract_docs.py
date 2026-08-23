"""Locked contract text lives in ADRs and the operator QA guide."""
from __future__ import annotations

from pathlib import Path

from src.baseball_analytics.fantasy import FANTASY_CARDS_RELPATH
from src.baseball_analytics.storage import REQUIRED_MANIFEST_FIELDS

ROOT = Path(__file__).resolve().parents[1]


def test_adr_layout_uses_runs_current_and_cards_jsonl() -> None:
    adr = (ROOT / "docs/adr/0001-shared-artifact-contract.md").read_text(encoding="utf-8")
    assert "runs/{run_id}/" in adr
    assert "current/" in adr
    assert FANTASY_CARDS_RELPATH in adr
    assert "fantasy_cards_{as_of_date}.json" in adr  # named as the voided path
    assert "Not `fantasy/fantasy_cards_{as_of_date}.json`" in adr or "Not `fantasy/fantasy_cards_" in adr
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


def test_qa_guide_documents_file_uri_how_to_verify() -> None:
    guide = (ROOT / "docs/shared_artifacts.md").read_text(encoding="utf-8")
    assert "ARTIFACTS_URI=file://" in guide
    assert "file:///tmp/btee-qa" in guide
    assert "current/fantasy/cards.jsonl" in guide
    assert "test ! -e /tmp/btee-qa/current/fantasy/fantasy_cards_2026-08-23.json" in guide
    for field in REQUIRED_MANIFEST_FIELDS:
        assert field in guide
