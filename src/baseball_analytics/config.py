from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path

import yaml

DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_PARTITION_LEAGUE = "mlb"
DEFAULT_PARTITION_LEVEL = "mlb"
DEFAULT_CACHE_TTL_S = 300


def load_settings(config_path: str = "config/settings.yaml") -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env_first(environ: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        raw = environ.get(name)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def resolve_artifacts_uri(
    settings: Mapping | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Return the configured shared-storage URI, or None when unset.

    Precedence: ``ARTIFACTS_URI`` env, then ``artifacts_uri`` in settings.
    Empty / missing values mean local ``artifacts/`` only.
    """
    env = os.environ if environ is None else environ
    raw = _env_first(env, "ARTIFACTS_URI")
    if raw is None and settings is not None:
        configured = settings.get("artifacts_uri")
        if configured is not None and str(configured).strip():
            raw = str(configured).strip()
    return raw


@dataclass(frozen=True)
class ArtifactSettings:
    """Vendor-agnostic artifact location (S3-compatible URI or local fallback)."""

    uri: str | None
    local_dir: Path
    league: str
    level: str
    cache_dir: Path
    cache_ttl_s: int = DEFAULT_CACHE_TTL_S


def load_artifact_settings(
    config_path: str = "config/settings.yaml",
    *,
    settings: Mapping | None = None,
    environ: Mapping[str, str] | None = None,
) -> ArtifactSettings:
    """Resolve artifact URI, partition, and local fallback from env + YAML."""
    data: Mapping
    if settings is not None:
        data = settings
    else:
        path = Path(config_path)
        data = load_settings(str(path)) if path.is_file() else {}

    env = os.environ if environ is None else environ
    partition = data.get("artifacts_partition") or {}
    league = (
        _env_first(env, "ARTIFACTS_LEAGUE")
        or str(partition.get("league") or DEFAULT_PARTITION_LEAGUE)
    )
    level = (
        _env_first(env, "ARTIFACTS_LEVEL")
        or str(partition.get("level") or DEFAULT_PARTITION_LEVEL)
    )
    local_dir = Path(data.get("artifacts_dir") or DEFAULT_ARTIFACTS_DIR)
    cache_dir = Path(data.get("artifacts_cache_dir") or (local_dir / ".remote_cache"))
    ttl_raw = _env_first(env, "ARTIFACTS_CACHE_TTL")
    if ttl_raw is not None:
        cache_ttl_s = int(ttl_raw)
    else:
        cache_ttl_s = int(data.get("artifacts_cache_ttl", DEFAULT_CACHE_TTL_S))
    return ArtifactSettings(
        uri=resolve_artifacts_uri(data, env),
        local_dir=local_dir,
        league=league.strip().lower(),
        level=level.strip().lower(),
        cache_dir=cache_dir,
        cache_ttl_s=cache_ttl_s,
    )
