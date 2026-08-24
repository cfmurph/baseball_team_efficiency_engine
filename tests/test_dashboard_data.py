"""Named dashboard loaders resolve via storage, never raw Path('artifacts')."""
from __future__ import annotations

import ast
from pathlib import Path

from src.baseball_analytics.config import ArtifactSettings
from dashboard.data import ARTIFACT_NAMES, METRICS_MANIFEST_NAME, resolve_file, resolve_metrics_manifest
from dashboard.state import SEASON_YEAR, SELECTED_LEAGUE, SELECTED_TEAM, SHARED_STATE_KEYS


def test_shared_session_state_keys() -> None:
    assert SEASON_YEAR == "season_year"
    assert SELECTED_TEAM == "selected_team"
    assert SELECTED_LEAGUE == "selected_league"
    assert SHARED_STATE_KEYS == (SEASON_YEAR, SELECTED_TEAM, SELECTED_LEAGUE, "nav_page")


def test_named_loaders_exist_and_pages_avoid_raw_paths() -> None:
    views_dir = Path("dashboard/views")
    for path in views_dir.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert 'Path("artifacts")' not in source
        assert "team_onfield_contract_metrics.csv" not in source
        assert "player_season_metrics.csv" not in source
        assert "src.baseball_analytics.storage" not in source
        assert "ARTIFACTS_URI" not in source


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


def test_app_keeps_resolve_file_and_load_helpers() -> None:
    tree = ast.parse(Path("dashboard/app.py").read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"_resolve_file", "_load", "page_league_snapshot", "page_player_explorer"} <= names
