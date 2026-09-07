"""Named dashboard loaders resolve via storage, never raw Path('artifacts')."""
from __future__ import annotations

import pytest

import ast
from pathlib import Path

from src.baseball_analytics.config import ArtifactSettings
from dashboard.data import (
    ARTIFACT_NAMES,
    METRICS_MANIFEST_NAME,
    load_metrics_manifest,
    load_named_artifact,
    resolve_file,
    resolve_metrics_manifest,
)
from dashboard.state import SEASON_YEAR, SELECTED_LEAGUE, SELECTED_TEAM, SHARED_STATE_KEYS

@pytest.mark.unit
def test_shared_session_state_keys() -> None:
    assert SEASON_YEAR == "season_year"
    assert SELECTED_TEAM == "selected_team"
    assert SELECTED_LEAGUE == "selected_league"
    assert SHARED_STATE_KEYS == (SEASON_YEAR, SELECTED_TEAM, SELECTED_LEAGUE, "nav_page")

@pytest.mark.unit
def test_named_loaders_exist_and_pages_avoid_raw_paths() -> None:
    views_dir = Path("dashboard/views")
    for path in views_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert 'Path("artifacts")' not in source
        assert "team_onfield_contract_metrics.csv" not in source
        assert "player_season_metrics.csv" not in source
        assert "src.baseball_analytics.storage" not in source
        assert "ARTIFACTS_URI" not in source

@pytest.mark.unit
def test_only_data_module_calls_resolve_artifact() -> None:
    dashboard = Path("dashboard")
    for path in dashboard.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if path.name == "data.py":
            assert "resolve_artifact" in source
            continue
        if path.name == "fantasy_app.py":
            # Separate BenchOrStart entrypoint; FO loaders stay in data.py.
            continue
        assert "resolve_artifact" not in source
        assert "from src.baseball_analytics.storage import" not in source

@pytest.mark.unit
def test_data_module_has_named_loaders() -> None:
    source = Path("dashboard/data.py").read_text(encoding="utf-8")
    for name in (
        "load_team_metrics",
        "load_player_season_metrics",
        "load_window_phases",
        "load_frontier_data",
        "load_win_model_metrics",
        "load_metrics_manifest",
    ):
        assert f"def {name}(" in source
    assert "resolve_artifact" in source
    assert ARTIFACT_NAMES["metrics"] == "team_onfield_contract_metrics.csv"
    assert ARTIFACT_NAMES["metrics_manifest"] == METRICS_MANIFEST_NAME
    assert METRICS_MANIFEST_NAME == "metrics_manifest.json"

@pytest.mark.integration
def test_resolve_file_uses_local_fallback(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("year_id\n2015\n")
    settings = ArtifactSettings(
        uri=None,
        local_dir=local,
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    path = resolve_file("metrics", settings)
    assert path == local / "team_onfield_contract_metrics.csv"
    assert resolve_file("players", settings) is None
    (local / METRICS_MANIFEST_NAME).write_text(
        '{"current_season_missing": true, "active_season": 2026, "seasons_present": [2024]}\n'
    )
    assert resolve_file("metrics_manifest", settings) == local / METRICS_MANIFEST_NAME
    manifest_path = resolve_metrics_manifest(settings)
    assert manifest_path == local / METRICS_MANIFEST_NAME

@pytest.mark.integration
def test_resolve_file_prefers_current_over_legacy_latest(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    current = shared / "current"
    latest = shared / "mlb" / "mlb" / "latest"
    current.mkdir(parents=True)
    latest.mkdir(parents=True)
    (current / "team_onfield_contract_metrics.csv").write_text("year_id\n2016\n")
    (latest / "team_onfield_contract_metrics.csv").write_text("year_id\n2015\n")
    settings = ArtifactSettings(
        uri=f"file://{shared}",
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    path = resolve_file("metrics", settings)
    assert path is not None
    assert path.read_text() == "year_id\n2016\n"

@pytest.mark.integration
def test_resolve_file_uses_shared_latest_when_uri_set(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    latest = shared / "mlb" / "mlb" / "latest"
    latest.mkdir(parents=True)
    (latest / "team_onfield_contract_metrics.csv").write_text("year_id\n2015\n")
    settings = ArtifactSettings(
        uri=f"file://{shared}",
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    path = resolve_file("metrics", settings)
    assert path is not None
    assert path.read_text() == "year_id\n2015\n"

@pytest.mark.unit
def test_load_named_artifact_rejects_unknown_and_json_keys() -> None:
    assert load_named_artifact("not_a_key") is None
    assert load_named_artifact("metrics_manifest") is None
    assert resolve_file("not_a_key") is None


@pytest.mark.unit
def test_load_metrics_manifest_returns_none_on_invalid_or_non_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import dashboard.data as data_mod

    monkeypatch.setattr(
        data_mod,
        "_read_json",
        lambda path_str: json.loads(Path(path_str).read_text(encoding="utf-8")),
    )

    monkeypatch.setattr(data_mod, "resolve_metrics_manifest", lambda: None)
    assert load_metrics_manifest() is None

    broken = tmp_path / "broken.json"
    broken.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(data_mod, "resolve_metrics_manifest", lambda: broken)
    assert load_metrics_manifest() is None

    array_payload = tmp_path / "array.json"
    array_payload.write_text("[2024, 2025]", encoding="utf-8")
    monkeypatch.setattr(data_mod, "resolve_metrics_manifest", lambda: array_payload)
    assert load_metrics_manifest() is None

    valid = tmp_path / "valid.json"
    valid.write_text(
        '{"current_season_missing": true, "active_season": 2026, "seasons_present": [2024]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(data_mod, "resolve_metrics_manifest", lambda: valid)
    payload = load_metrics_manifest()
    assert payload == {
        "current_season_missing": True,
        "active_season": 2026,
        "seasons_present": [2024],
    }


@pytest.mark.unit
def test_load_named_artifact_reads_csv_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    import dashboard.data as data_mod

    csv_path = tmp_path / "team_onfield_contract_metrics.csv"
    csv_path.write_text("year_id,team_id\n2015,NYY\n", encoding="utf-8")
    monkeypatch.setattr(data_mod, "_read_csv", lambda path_str: pd.read_csv(path_str))
    monkeypatch.setattr(data_mod, "resolve_file", lambda key, settings=None: csv_path)
    frame = load_named_artifact("metrics")
    assert frame is not None
    assert list(frame["year_id"]) == [2015]
    assert list(frame["team_id"]) == ["NYY"]

    monkeypatch.setattr(data_mod, "resolve_file", lambda key, settings=None: None)
    assert load_named_artifact("players") is None


@pytest.mark.unit
def test_app_keeps_resolve_file_and_load_helpers() -> None:
    tree = ast.parse(Path("dashboard/app.py").read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"_resolve_file", "_load", "page_league_snapshot", "page_player_explorer"} <= names
