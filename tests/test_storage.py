"""URI resolution, runs/current layout, upload, fallback, and manifest fields."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.fantasy import FANTASY_CARDS_RELPATH
from src.baseball_analytics.storage import (
    REQUIRED_MANIFEST_FIELDS,
    ArtifactUploadError,
    FileBackend,
    S3Backend,
    artifact_source_label,
    classify_artifact_relpath,
    current_object_key,
    default_as_of_date,
    default_run_date,
    default_run_id,
    object_key,
    parse_artifacts_uri,
    partition_key,
    publish_nightly_artifacts,
    remote_lookup_keys,
    resolve_artifact,
    resolve_named_artifacts,
    run_object_key,
    upload_artifacts,
)


class MemoryBackend:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.fail_get = False
        self.fail_put_prefix: str | None = None

    def put(self, relative_key: str, data: bytes) -> None:
        if self.fail_put_prefix and relative_key.startswith(self.fail_put_prefix):
            raise RuntimeError("remote write failed")
        self.objects[relative_key] = data

    def get(self, relative_key: str) -> bytes | None:
        self.gets.append(relative_key)
        if self.fail_get:
            raise RuntimeError("remote unreachable")
        return self.objects.get(relative_key)


def _settings(tmp_path: Path, **overrides) -> ArtifactSettings:
    defaults = dict(
        uri="memory://unused",
        local_dir=tmp_path / "artifacts",
        league="mlb",
        level="mlb",
        cache_dir=tmp_path / "cache",
        cache_ttl_s=0,
    )
    defaults.update(overrides)
    return ArtifactSettings(**defaults)


def _seed_local(tmp_path: Path) -> Path:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("year_id\n2015\n")
    (local / "win_model_metrics.csv").write_text("model,mae\nlr,3\n")
    (local / "win_model_actual_vs_predicted.png").write_bytes(b"png")
    (local / "player_season_metrics.csv").write_text(
        "player_id,war_source\njudgeaa01,real\ntroutmi01,approx\n"
    )
    (local / ".remote_cache").mkdir()
    (local / ".remote_cache" / "stale.csv").write_text("nope\n")
    return local


def test_parse_s3_uri_with_and_without_prefix() -> None:
    bare = parse_artifacts_uri("s3://metrics-bucket")
    assert bare.scheme == "s3"
    assert bare.bucket == "metrics-bucket"
    assert bare.prefix == ""

    prefixed = parse_artifacts_uri("s3://metrics-bucket/prod/baseball/")
    assert prefixed.bucket == "metrics-bucket"
    assert prefixed.prefix == "prod/baseball"

    alias = parse_artifacts_uri("s3a://metrics-bucket/data")
    assert alias.scheme == "s3"
    assert alias.prefix == "data"


def test_parse_r2_and_gs_schemes() -> None:
    r2 = parse_artifacts_uri("r2://my-r2/baseball")
    assert r2.scheme == "s3"
    assert r2.bucket == "my-r2"
    assert r2.prefix == "baseball"

    gs = parse_artifacts_uri("gs://lake/prefix")
    assert gs.scheme == "gs"
    assert gs.bucket == "lake"
    assert gs.prefix == "prefix"


def test_parse_file_and_bare_paths(tmp_path: Path) -> None:
    parsed = parse_artifacts_uri("file:///shared/artifacts")
    assert parsed.scheme == "file"
    assert parsed.prefix == "/shared/artifacts"

    bare = parse_artifacts_uri(str(tmp_path / "store"))
    assert bare.scheme == "file"
    assert bare.prefix == str(tmp_path / "store")


def test_parse_rejects_empty_and_unknown_schemes() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_artifacts_uri("  ")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_artifacts_uri("https://example.com/bucket")
    with pytest.raises(ValueError, match="bucket"):
        parse_artifacts_uri("s3:///no-bucket")


def test_legacy_partition_and_object_key_are_league_level_date() -> None:
    assert partition_key("MLB", "mlb", "2026-08-23") == "mlb/mlb/2026-08-23"
    assert partition_key("mlb", "mlb", "latest") == "mlb/mlb/latest"
    assert (
        object_key("mlb", "mlb", "2026-08-23", "team_onfield_contract_metrics.csv")
        == "mlb/mlb/2026-08-23/team_onfield_contract_metrics.csv"
    )
    with pytest.raises(ValueError, match="run_date"):
        partition_key("mlb", "mlb", "08-23-2026")
    with pytest.raises(ValueError, match="league"):
        partition_key("mlb/extra", "mlb", "2026-08-23")


def test_run_and_current_object_keys() -> None:
    assert run_object_key("20260823T080012Z", "metrics/a.csv") == (
        "runs/20260823T080012Z/metrics/a.csv"
    )
    assert current_object_key(FANTASY_CARDS_RELPATH) == "current/fantasy/cards.jsonl"
    with pytest.raises(ValueError, match="run_id"):
        run_object_key("current", "metrics/a.csv")
    with pytest.raises(ValueError, match="run_id"):
        run_object_key("runs/nested", "metrics/a.csv")


def test_classify_keeps_csv_names_and_cards_path() -> None:
    assert classify_artifact_relpath("player_season_metrics.csv") == (
        "metrics/player_season_metrics.csv"
    )
    assert classify_artifact_relpath("win_model_actual_vs_predicted.png") == (
        "models/win_model_actual_vs_predicted.png"
    )
    assert classify_artifact_relpath("cards.jsonl") == FANTASY_CARDS_RELPATH
    assert classify_artifact_relpath("fantasy/cards.jsonl") == FANTASY_CARDS_RELPATH


def test_default_as_of_and_run_id_use_env_overrides() -> None:
    assert default_as_of_date(environ={"ARTIFACTS_AS_OF_DATE": "2024-07-04"}) == "2024-07-04"
    assert default_run_date(environ={"ARTIFACTS_RUN_DATE": "2024-07-04"}) == "2024-07-04"
    assert default_run_id(environ={"ARTIFACTS_RUN_ID": "20240704T120000Z"}) == "20240704T120000Z"
    assert default_run_id(environ={"GITHUB_RUN_ID": "987654321"}) == "987654321"


def test_env_uri_overrides_settings_yaml() -> None:
    settings = load_artifact_settings(
        settings={
            "artifacts_uri": "s3://from-yaml/prefix",
            "artifacts_dir": "artifacts",
            "artifacts_partition": {"league": "mlb", "level": "aaa"},
        },
        environ={
            "ARTIFACTS_URI": "s3://from-env/other",
            "ARTIFACTS_LEVEL": "aa",
        },
    )
    assert settings.uri == "s3://from-env/other"
    assert settings.league == "mlb"
    assert settings.level == "aa"


def test_blank_uri_means_local_only() -> None:
    settings = load_artifact_settings(
        settings={"artifacts_uri": "  ", "artifacts_dir": "artifacts"},
        environ={},
    )
    assert settings.uri is None


def test_upload_writes_runs_and_current_not_latest(tmp_path: Path) -> None:
    local = _seed_local(tmp_path)
    backend = MemoryBackend()
    result = upload_artifacts(
        local,
        _settings(tmp_path, uri="s3://bucket/prefix"),
        run_id="20260823T080012Z",
        as_of_date="2026-08-23",
        backend=backend,
        pipeline_steps=["pull_sources", "build_metrics"],
        git_sha="abc123",
    )

    assert result.skipped is False
    assert result.current_updated is True
    assert result.relative_prefix == "runs/20260823T080012Z"
    assert result.as_of_date == "2026-08-23"
    assert "metrics/team_onfield_contract_metrics.csv" in result.files
    assert FANTASY_CARDS_RELPATH in result.files
    assert "metrics/team_onfield_contract_metrics.csv" in result.files
    assert "models/win_model_actual_vs_predicted.png" in result.files

    run_metrics = "runs/20260823T080012Z/metrics/team_onfield_contract_metrics.csv"
    assert run_metrics in backend.objects
    assert "current/metrics/team_onfield_contract_metrics.csv" in backend.objects
    assert f"runs/20260823T080012Z/{FANTASY_CARDS_RELPATH}" in backend.objects
    assert f"current/{FANTASY_CARDS_RELPATH}" in backend.objects
    assert "mlb/mlb/latest/team_onfield_contract_metrics.csv" not in backend.objects
    assert not any("fantasy_cards_" in key for key in backend.objects)

    manifest = json.loads(backend.objects["current/manifest.json"].decode())
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(manifest)
    assert manifest["schema_version"] == "1.0"
    assert manifest["as_of_date"] == "2026-08-23"
    assert manifest["git_sha"] == "abc123"
    assert manifest["pipeline_steps"] == ["pull_sources", "build_metrics"]
    assert manifest["war_source_summary"]["bbref"] == 1
    assert manifest["war_source_summary"]["approx"] == 1
    assert FANTASY_CARDS_RELPATH in manifest["files"]
    assert "stale.csv" not in manifest["files"]


def test_upload_refuses_to_mutate_existing_run(tmp_path: Path) -> None:
    local = _seed_local(tmp_path)
    backend = MemoryBackend()
    settings = _settings(tmp_path, uri="s3://bucket/prefix")
    upload_artifacts(
        local,
        settings,
        run_id="20260823T080012Z",
        as_of_date="2026-08-23",
        backend=backend,
    )
    (local / "team_onfield_contract_metrics.csv").write_text("mutated\n")
    with pytest.raises(ArtifactUploadError, match="immutable"):
        upload_artifacts(
            local,
            settings,
            run_id="20260823T080012Z",
            as_of_date="2026-08-23",
            backend=backend,
        )
    stored = backend.objects["runs/20260823T080012Z/metrics/team_onfield_contract_metrics.csv"]
    assert stored == b"year_id\n2015\n"


def test_failed_run_write_does_not_update_current(tmp_path: Path) -> None:
    local = _seed_local(tmp_path)
    backend = MemoryBackend()
    backend.fail_put_prefix = "runs/"
    with pytest.raises(ArtifactUploadError):
        upload_artifacts(
            local,
            _settings(tmp_path, uri="s3://bucket"),
            run_id="20260823T080012Z",
            backend=backend,
        )
    assert not any(key.startswith("current/") for key in backend.objects)


def test_upload_skipped_when_uri_unset(tmp_path: Path) -> None:
    result = upload_artifacts(tmp_path, _settings(tmp_path, uri=None), as_of_date="2026-08-23")
    assert result.skipped is True
    assert result.reason == "no_uri"


def test_upload_fails_when_local_dir_empty(tmp_path: Path) -> None:
    empty = tmp_path / "artifacts"
    empty.mkdir()
    with pytest.raises(ArtifactUploadError, match="No artifact files"):
        upload_artifacts(
            empty,
            _settings(tmp_path),
            as_of_date="2026-08-23",
            backend=MemoryBackend(),
        )


def test_resolve_prefers_remote_current_over_local(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "metrics.csv").write_text("local\n")

    backend = MemoryBackend()
    backend.put("current/metrics/metrics.csv", b"remote\n")

    path = resolve_artifact("metrics.csv", _settings(tmp_path), backend=backend)
    assert path is not None
    assert path.read_text() == "remote\n"
    assert "current" in path.parts


def test_resolve_compat_bridge_reads_legacy_latest(tmp_path: Path) -> None:
    backend = MemoryBackend()
    backend.put("mlb/mlb/latest/metrics.csv", b"legacy\n")
    path = resolve_artifact("metrics.csv", _settings(tmp_path), backend=backend)
    assert path is not None
    assert path.read_text() == "legacy\n"
    assert any(key == "mlb/mlb/latest/metrics.csv" for key in remote_lookup_keys("metrics.csv", _settings(tmp_path)))


def test_resolve_falls_back_to_local_when_uri_unset(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "metrics.csv").write_text("local-only\n")

    path = resolve_artifact("metrics.csv", _settings(tmp_path, uri=None))
    assert path == local / "metrics.csv"
    assert path.read_text() == "local-only\n"


def test_resolve_falls_back_to_local_when_remote_unreachable(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "metrics.csv").write_text("local-fallback\n")

    backend = MemoryBackend()
    backend.fail_get = True

    path = resolve_artifact(
        "metrics.csv",
        _settings(tmp_path, uri="s3://bucket/prefix"),
        backend=backend,
    )
    assert path == local / "metrics.csv"
    assert path.read_text() == "local-fallback\n"


def test_resolve_uses_stale_cache_when_remote_fails_and_local_missing(tmp_path: Path) -> None:
    cache = tmp_path / "cache" / "current" / "metrics"
    cache.mkdir(parents=True)
    cached = cache / "metrics.csv"
    cached.write_text("cached\n")

    backend = MemoryBackend()
    backend.fail_get = True

    path = resolve_artifact(
        "metrics.csv",
        _settings(tmp_path, uri="s3://bucket/prefix", cache_dir=tmp_path / "cache"),
        backend=backend,
    )
    assert path == cached


def test_resolve_returns_none_when_everywhere_missing(tmp_path: Path) -> None:
    backend = MemoryBackend()
    assert (
        resolve_artifact("missing.csv", _settings(tmp_path, uri="s3://bucket"), backend=backend)
        is None
    )


def test_resolve_named_artifacts_maps_keys(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("ok\n")
    resolved = resolve_named_artifacts(
        {"metrics": local / "team_onfield_contract_metrics.csv", "players": "missing.csv"},
        _settings(tmp_path, uri=None),
    )
    assert resolved["metrics"] == local / "team_onfield_contract_metrics.csv"
    assert resolved["players"] is None


def test_resolve_nested_current_fantasy_cards_jsonl(tmp_path: Path) -> None:
    backend = MemoryBackend()
    backend.put("current/fantasy/cards.jsonl", b'{"recommendation_type":"start"}\n')
    path = resolve_artifact(
        "current/fantasy/cards.jsonl",
        _settings(tmp_path, uri="s3://bucket/prefix"),
        backend=backend,
    )
    assert path is not None
    assert path.read_text() == '{"recommendation_type":"start"}\n'
    assert backend.gets[0] == "current/fantasy/cards.jsonl"
    assert not any("fantasy_cards_" in key for key in backend.gets)


def test_resolve_nested_local_current_fantasy_cards_jsonl(tmp_path: Path) -> None:
    lake = tmp_path / "artifacts" / "current" / "fantasy"
    lake.mkdir(parents=True)
    (lake / "cards.jsonl").write_text("local-cards\n", encoding="utf-8")
    path = resolve_artifact("current/fantasy/cards.jsonl", _settings(tmp_path, uri=None))
    assert path == lake / "cards.jsonl"


def test_file_uri_qa_layout_and_fallback(tmp_path: Path) -> None:
    """How-to-verify with ARTIFACTS_URI=file:// (docs/shared_artifacts.md)."""
    local = _seed_local(tmp_path)
    shared = tmp_path / "btee-qa"
    settings = _settings(tmp_path, uri=f"file://{shared}")

    result = upload_artifacts(
        local,
        settings,
        run_id="qa20260823T000000Z",
        as_of_date="2026-08-23",
        git_sha="deadbeef",
    )
    assert result.current_updated is True

    run_root = shared / "runs" / "qa20260823T000000Z"
    current = shared / "current"
    assert (run_root / "manifest.json").is_file()
    assert (run_root / "metrics" / "team_onfield_contract_metrics.csv").is_file()
    assert (run_root / "fantasy" / "cards.jsonl").is_file()
    assert (current / "manifest.json").is_file()
    assert (current / "fantasy" / "cards.jsonl").is_file()
    assert not (current / "fantasy" / "fantasy_cards_2026-08-23.json").exists()
    assert not (shared / "mlb").exists()

    manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(manifest)
    assert manifest["schema_version"] == "1.0"
    assert manifest["as_of_date"] == "2026-08-23"

    (local / "team_onfield_contract_metrics.csv").unlink()
    path = resolve_artifact("team_onfield_contract_metrics.csv", settings)
    assert path is not None
    assert path.read_text() == "year_id\n2015\n"
    assert artifact_source_label(settings) == "remote"

    fallback = resolve_artifact(
        "win_model_metrics.csv",
        _settings(tmp_path, uri=None),
    )
    assert fallback == local / "win_model_metrics.csv"
    assert artifact_source_label(_settings(tmp_path, uri=None)) == "local"


def test_file_backend_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    backend = FileBackend(root)
    key = current_object_key("metrics/metrics.csv")
    backend.put(key, b"from-file\n")
    assert backend.get(key) == b"from-file\n"
    assert backend.get("current/metrics/missing.csv") is None
    assert (root / key).read_bytes() == b"from-file\n"


def test_s3_backend_uses_prefix_and_maps_404() -> None:
    stored: dict[str, bytes] = {}

    class FakeClient:
        def put_object(self, Bucket, Key, Body):
            stored[(Bucket, Key)] = Body

        def get_object(self, Bucket, Key):
            if (Bucket, Key) not in stored:
                raise SimpleNamespaceError()
            return {"Body": SimpleNamespace(read=lambda: stored[(Bucket, Key)])}

    class SimpleNamespaceError(Exception):
        response = {"Error": {"Code": "NoSuchKey"}}

    backend = S3Backend("bucket", "prod/baseball", FakeClient())
    backend.put("current/metrics/metrics.csv", b"s3\n")
    assert stored[("bucket", "prod/baseball/current/metrics/metrics.csv")] == b"s3\n"
    assert backend.get("current/metrics/metrics.csv") == b"s3\n"
    assert backend.get("current/metrics/missing.csv") is None


def test_artifact_source_badge_remote_local_missing(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("ok\n")

    backend = MemoryBackend()
    backend.put("current/manifest.json", b"{}")
    remote = _settings(tmp_path, uri="s3://metrics-bucket/prod")
    assert artifact_source_label(remote, backend=backend) == "remote"

    assert artifact_source_label(_settings(tmp_path, uri=None)) == "local"

    empty = tmp_path / "empty"
    empty.mkdir()
    missing = _settings(tmp_path, uri=None, local_dir=empty)
    assert artifact_source_label(missing) == "missing"

    backend.fail_get = True
    assert artifact_source_label(remote, backend=backend) == "local"


def test_publish_nightly_skips_without_uri(tmp_path: Path) -> None:
    result = publish_nightly_artifacts(
        settings=_settings(tmp_path, uri=None),
        as_of_date="2026-08-23",
    )
    assert result.skipped is True
