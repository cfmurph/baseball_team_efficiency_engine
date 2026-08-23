"""Shared artifact storage (S3-compatible) with a local ``artifacts/`` fallback.

Object layout under the configured URI prefix::

    {league}/{level}/{run_date}/<relative-file>
    {league}/{level}/latest/<relative-file>
    {league}/{level}/latest/manifest.json

``<relative-file>`` is any path under local ``artifacts/`` (flat CSVs today;
nested files are first-class). ``run_date`` is ``YYYY-MM-DD`` (UTC unless
``ARTIFACTS_RUN_DATE`` is set). ``latest/`` is a copy of the most recent
successful publish so dashboards do not need to list history. Future
minor-league feeds use the same shape (e.g. ``milb/aaa/2026-08-23/``).
Reserved (unpublished in this slice): ``fantasy/cards.jsonl``.
See ``docs/adr/0001-shared-artifact-layout.md``.

URI schemes:
- ``s3://bucket/optional-prefix`` — AWS S3 or any S3-compatible store (R2 via
  ``AWS_ENDPOINT_URL``)
- ``file:///absolute/path`` or a bare filesystem path — shared disk / tests
- unset / empty — local ``artifacts_dir`` only
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Protocol
from urllib.parse import urlparse

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings

log = logging.getLogger(__name__)

LATEST_LABEL = "latest"
MANIFEST_NAME = "manifest.json"
SKIP_DIR_NAMES = {".remote_cache", ".cache", "__pycache__"}


class ArtifactStoreError(RuntimeError):
    """Base error for shared artifact storage."""


class ArtifactUploadError(ArtifactStoreError):
    """Raised when a configured remote upload fails."""


@dataclass(frozen=True)
class ParsedURI:
    scheme: str
    bucket: str | None
    prefix: str
    raw: str


class ArtifactBackend(Protocol):
    def put(self, relative_key: str, data: bytes) -> None: ...

    def get(self, relative_key: str) -> bytes | None: ...


@dataclass
class UploadResult:
    uri: str
    run_date: str
    relative_prefix: str
    files: list[str]
    skipped: bool = False
    reason: str | None = None


def parse_artifacts_uri(uri: str) -> ParsedURI:
    """Parse an S3-compatible or filesystem artifacts URI."""
    text = str(uri).strip()
    if not text:
        raise ValueError("artifacts URI is empty")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"s3", "s3a"}:
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"S3 URI is missing a bucket: {uri!r}")
        prefix = parsed.path.lstrip("/").rstrip("/")
        return ParsedURI(scheme="s3", bucket=bucket, prefix=prefix, raw=text)
    if scheme == "file":
        path = parsed.path
        if parsed.netloc and parsed.netloc not in {"localhost", ""}:
            path = f"/{parsed.netloc}{parsed.path}"
        if not path:
            raise ValueError(f"file URI is missing a path: {uri!r}")
        return ParsedURI(scheme="file", bucket=None, prefix=str(Path(path)), raw=text)
    if scheme == "":
        return ParsedURI(scheme="file", bucket=None, prefix=str(Path(text)), raw=text)
    raise ValueError(
        f"Unsupported artifacts URI scheme {scheme!r}. "
        "Use s3://bucket/prefix or file:///path (see docs/shared_artifacts.md)."
    )


def partition_key(league: str, level: str, run_date: str) -> str:
    """Return ``{league}/{level}/{run_date}`` after validating each segment."""
    return "/".join(
        (
            _partition_token(league, "league"),
            _partition_token(level, "level"),
            _partition_token(run_date, "run_date", allow_latest=True),
        )
    )


def relative_artifact_key(filename: str | Path) -> str:
    """Return a partition-relative key (``foo.csv`` or ``dir/foo.csv``).

    Absolute paths keep only the basename so existing dashboard ``Path``
    objects keep working. Relative paths keep subdirectories so a later
    file such as ``fantasy/cards.jsonl`` needs no URI redesign.
    """
    text = str(filename).replace("\\", "/").strip()
    if not text:
        raise ValueError("artifact filename is empty")
    path = Path(text)
    rel = path.name if path.is_absolute() else text.lstrip("/")
    if rel.startswith("./"):
        rel = rel[2:]
    if not rel or rel.endswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"Invalid artifact filename: {filename!r}")
    return rel


def object_key(league: str, level: str, run_date: str, filename: str) -> str:
    return f"{partition_key(league, level, run_date)}/{relative_artifact_key(filename)}"


def default_run_date(
    *,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    override = env.get("ARTIFACTS_RUN_DATE", "").strip()
    if override:
        return _partition_token(override, "run_date")
    current = now or datetime.now(timezone.utc)
    return current.date().isoformat()


def artifact_source_label(settings: ArtifactSettings) -> str:
    if not settings.uri:
        return "local"
    parsed = parse_artifacts_uri(settings.uri)
    if parsed.scheme == "s3":
        suffix = f"/{parsed.prefix}" if parsed.prefix else ""
        return f"shared s3://{parsed.bucket}{suffix}"
    return "shared filesystem"


def iter_artifact_files(local_dir: Path) -> list[Path]:
    if not local_dir.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(local_dir).parts
        if any(part in SKIP_DIR_NAMES or part.startswith(".") for part in relative_parts):
            continue
        files.append(path)
    return files


class FileBackend:
    """Shared filesystem rooted at a ``file://`` or bare path URI."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def put(self, relative_key: str, data: bytes) -> None:
        dest = self.root / relative_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def get(self, relative_key: str) -> bytes | None:
        path = self.root / relative_key
        if not path.is_file():
            return None
        return path.read_bytes()


class S3Backend:
    """S3-compatible object store (AWS S3, Cloudflare R2, MinIO, …)."""

    def __init__(self, bucket: str, prefix: str, client: object) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self.client = client

    def _abs_key(self, relative_key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{relative_key}"
        return relative_key

    def put(self, relative_key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._abs_key(relative_key), Body=data)

    def get(self, relative_key: str) -> bytes | None:
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=self._abs_key(relative_key)
            )
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise
        body = response["Body"]
        return body.read() if hasattr(body, "read") else body


def build_s3_client(environ: Mapping[str, str] | None = None) -> object:
    """Build a boto3 S3 client from standard AWS / R2 environment variables."""
    import boto3

    env = os.environ if environ is None else environ
    kwargs: dict[str, str] = {}
    endpoint = env.get("AWS_ENDPOINT_URL", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = (
        env.get("AWS_REGION", "").strip()
        or env.get("AWS_DEFAULT_REGION", "").strip()
        or "us-east-1"
    )
    kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def open_backend(
    uri: str,
    *,
    s3_client: object | None = None,
    environ: Mapping[str, str] | None = None,
) -> ArtifactBackend:
    parsed = parse_artifacts_uri(uri)
    if parsed.scheme == "s3":
        client = s3_client if s3_client is not None else build_s3_client(environ)
        return S3Backend(parsed.bucket or "", parsed.prefix, client)
    return FileBackend(Path(parsed.prefix))


def upload_artifacts(
    local_dir: str | Path,
    settings: ArtifactSettings,
    *,
    run_date: str | None = None,
    backend: ArtifactBackend | None = None,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
) -> UploadResult:
    """Publish local artifact files to ``{league}/{level}/{run_date}`` and ``latest``."""
    if not settings.uri:
        return UploadResult(
            uri="",
            run_date=run_date or default_run_date(now=now, environ=environ),
            relative_prefix="",
            files=[],
            skipped=True,
            reason="no_uri",
        )

    source = Path(local_dir)
    files = iter_artifact_files(source)
    if not files:
        raise ArtifactUploadError(
            f"No artifact files to upload from {source}. "
            "Run the pipeline before publishing."
        )

    dated = run_date or default_run_date(now=now, environ=environ)
    store = backend if backend is not None else open_backend(settings.uri, environ=environ)
    relative_names = [path.relative_to(source).as_posix() for path in files]
    created = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = {
        "league": settings.league,
        "level": settings.level,
        "run_date": dated,
        "created_at": created,
        "files": relative_names,
    }
    payload = json.dumps(manifest, indent=2).encode("utf-8")

    try:
        for dest in (dated, LATEST_LABEL):
            for path, name in zip(files, relative_names, strict=True):
                store.put(object_key(settings.league, settings.level, dest, name), path.read_bytes())
            store.put(
                object_key(settings.league, settings.level, dest, MANIFEST_NAME),
                payload,
            )
    except ArtifactStoreError:
        raise
    except Exception as exc:
        raise ArtifactUploadError(
            f"Failed to upload artifacts to {settings.uri}: {exc}"
        ) from exc

    relative_prefix = partition_key(settings.league, settings.level, dated)
    log.info(
        "Uploaded %d artifact files to %s/%s and %s/%s",
        len(relative_names),
        settings.uri.rstrip("/"),
        relative_prefix,
        settings.uri.rstrip("/"),
        partition_key(settings.league, settings.level, LATEST_LABEL),
    )
    return UploadResult(
        uri=settings.uri,
        run_date=dated,
        relative_prefix=relative_prefix,
        files=relative_names,
    )


def resolve_artifact(
    filename: str,
    settings: ArtifactSettings | None = None,
    *,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return a local path for ``filename``, preferring shared storage.

    Order: fresh remote cache → remote ``latest/`` (written to cache) →
    stale remote cache → local ``artifacts_dir`` → ``None``.
    Remote errors are logged and treated as a miss so the dashboard still
    loads from disk when the store is unreachable.
    """
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    try:
        name = relative_artifact_key(filename)
    except ValueError:
        return None
    local_path = cfg.local_dir / name
    cache_path = cfg.cache_dir / cfg.league / cfg.level / LATEST_LABEL / name

    if cfg.uri:
        if _cache_is_fresh(cache_path, cfg.cache_ttl_s):
            return cache_path
        try:
            store = backend if backend is not None else open_backend(cfg.uri, environ=environ)
            data = store.get(object_key(cfg.league, cfg.level, LATEST_LABEL, name))
        except Exception as exc:
            log.warning(
                "Remote artifact %s unavailable (%s); falling back to local",
                name,
                exc,
            )
        else:
            if data is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(data)
                return cache_path
        if cache_path.is_file():
            return cache_path

    if local_path.is_file():
        return local_path
    return None


def resolve_named_artifacts(
    files: Mapping[str, str | Path],
    settings: ArtifactSettings | None = None,
    *,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path | None]:
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    resolved: dict[str, Path | None] = {}
    for key, filename in files.items():
        resolved[key] = resolve_artifact(filename, cfg, backend=backend, environ=environ)
    return resolved


def publish_nightly_artifacts(
    config_path: str = "config/settings.yaml",
    *,
    settings: ArtifactSettings | None = None,
    backend: ArtifactBackend | None = None,
    run_date: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> UploadResult:
    """Upload ``artifacts_dir`` after a successful nightly pipeline run."""
    cfg = settings if settings is not None else load_artifact_settings(config_path, environ=environ)
    if not cfg.uri:
        log.info("ARTIFACTS_URI unset; skipping shared-storage upload")
        return UploadResult(
            uri="",
            run_date=run_date or default_run_date(environ=environ),
            relative_prefix="",
            files=[],
            skipped=True,
            reason="no_uri",
        )
    return upload_artifacts(
        cfg.local_dir,
        cfg,
        run_date=run_date,
        backend=backend,
        environ=environ,
    )


def _partition_token(value: str, name: str, *, allow_latest: bool = False) -> str:
    text = str(value).strip()
    if name in {"league", "level"}:
        text = text.lower()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid {name}: {value!r}")
    if name == "run_date" and text != LATEST_LABEL:
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid run_date {value!r}; expected YYYY-MM-DD") from exc
    if name == "run_date" and text == LATEST_LABEL and not allow_latest:
        raise ValueError("run_date cannot be 'latest' unless explicitly allowed")
    return text


def _cache_is_fresh(path: Path, ttl_s: int) -> bool:
    if ttl_s <= 0 or not path.is_file():
        return False
    return (time.time() - path.stat().st_mtime) < ttl_s


def _is_not_found(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = str(response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return True
    name = type(exc).__name__
    return name in {"NoSuchKey", "404"}
