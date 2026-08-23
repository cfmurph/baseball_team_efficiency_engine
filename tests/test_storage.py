"""URI resolution, partition layout, upload, and local fallback."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.storage import (
    ArtifactUploadError,
    FileBackend,
    S3Backend,
    artifact_source_label,
    default_run_date,
    object_key,
    parse_artifacts_uri,
    partition_key,
    publish_nightly_artifacts,
    resolve_artifact,
    resolve_named_artifacts,
    upload_artifacts,
)


class MemoryBackend:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.gets: list[str] = []
        self.fail_get = False

    def put(self, relative_key: str, data: bytes) -> None:
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


def test_partition_and_object_key_are_league_level_date() -> None:
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


def test_default_run_date_uses_env_override() -> None:
    assert default_run_date(environ={"ARTIFACTS_RUN_DATE": "2024-07-04"}) == "2024-07-04"


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


def test_upload_writes_dated_and_latest_partitions(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("year_id\n2015\n")
    (local / "win_model_metrics.csv").write_text("model,mae\nlr,3\n")
    (local / ".remote_cache").mkdir()
    (local / ".remote_cache" / "stale.csv").write_text("nope\n")

    backend = MemoryBackend()
    result = upload_artifacts(
        local,
        _settings(tmp_path, uri="s3://bucket/prefix"),
        run_date="2026-08-23",
        backend=backend,
    )

    assert result.skipped is False
    assert result.relative_prefix == "mlb/mlb/2026-08-23"
    assert result.files == [
        "team_onfield_contract_metrics.csv",
        "win_model_metrics.csv",
    ]
    assert "mlb/mlb/2026-08-23/team_onfield_contract_metrics.csv" in backend.objects
    assert "mlb/mlb/latest/team_onfield_contract_metrics.csv" in backend.objects
    assert "mlb/mlb/latest/manifest.json" in backend.objects
    assert "mlb/mlb/2026-08-23/stale.csv" not in backend.objects
    manifest = backend.objects["mlb/mlb/latest/manifest.json"].decode()
    assert "2026-08-23" in manifest
    assert "team_onfield_contract_metrics.csv" in manifest


def test_upload_skipped_when_uri_unset(tmp_path: Path) -> None:
    result = upload_artifacts(tmp_path, _settings(tmp_path, uri=None), run_date="2026-08-23")
    assert result.skipped is True
    assert result.reason == "no_uri"


def test_upload_fails_when_local_dir_empty(tmp_path: Path) -> None:
    empty = tmp_path / "artifacts"
    empty.mkdir()
    with pytest.raises(ArtifactUploadError, match="No artifact files"):
        upload_artifacts(empty, _settings(tmp_path), run_date="2026-08-23", backend=MemoryBackend())


def test_resolve_prefers_remote_latest_over_local(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "metrics.csv").write_text("local\n")

    backend = MemoryBackend()
    backend.put("mlb/mlb/latest/metrics.csv", b"remote\n")

    path = resolve_artifact("metrics.csv", _settings(tmp_path), backend=backend)
    assert path is not None
    assert path.read_text() == "remote\n"
    assert path.parent.name == "latest"


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
    cache = tmp_path / "cache" / "mlb" / "mlb" / "latest"
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


def test_file_uri_upload_then_resolve_without_local_copy(tmp_path: Path) -> None:
    local = tmp_path / "artifacts"
    local.mkdir()
    (local / "team_onfield_contract_metrics.csv").write_text("parity,1\n")
    shared = tmp_path / "shared"
    settings = _settings(tmp_path, uri=f"file://{shared}")

    upload_artifacts(local, settings, run_date="2026-08-23")
    (local / "team_onfield_contract_metrics.csv").unlink()

    path = resolve_artifact("team_onfield_contract_metrics.csv", settings)
    assert path is not None
    assert path.read_text() == "parity,1\n"
    assert (shared / "mlb" / "mlb" / "latest" / "team_onfield_contract_metrics.csv").is_file()
    assert (shared / "mlb" / "mlb" / "2026-08-23" / "manifest.json").is_file()


def test_file_backend_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    backend = FileBackend(root)
    key = object_key("mlb", "mlb", "latest", "metrics.csv")
    backend.put(key, b"from-file\n")
    assert backend.get(key) == b"from-file\n"
    assert backend.get("mlb/mlb/latest/missing.csv") is None
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
    backend.put("mlb/mlb/latest/metrics.csv", b"s3\n")
    assert stored[("bucket", "prod/baseball/mlb/mlb/latest/metrics.csv")] == b"s3\n"
    assert backend.get("mlb/mlb/latest/metrics.csv") == b"s3\n"
    assert backend.get("mlb/mlb/latest/missing.csv") is None


def test_artifact_source_label() -> None:
    local = ArtifactSettings(
        uri=None,
        local_dir=Path("artifacts"),
        league="mlb",
        level="mlb",
        cache_dir=Path("artifacts/.remote_cache"),
    )
    assert artifact_source_label(local) == "local"
    shared = ArtifactSettings(
        uri="s3://metrics-bucket/prod",
        local_dir=Path("artifacts"),
        league="mlb",
        level="mlb",
        cache_dir=Path("artifacts/.remote_cache"),
    )
    assert artifact_source_label(shared) == "shared s3://metrics-bucket/prod"
    file_uri = ArtifactSettings(
        uri="file:///tmp/shared",
        local_dir=Path("artifacts"),
        league="mlb",
        level="mlb",
        cache_dir=Path("artifacts/.remote_cache"),
    )
    assert artifact_source_label(file_uri) == "shared filesystem"


def test_publish_nightly_skips_without_uri(tmp_path: Path) -> None:
    result = publish_nightly_artifacts(
        settings=_settings(tmp_path, uri=None),
        run_date="2026-08-23",
    )
    assert result.skipped is True
