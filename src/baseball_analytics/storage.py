"""Shared artifact storage with a vendor-agnostic URI and local fallback.

Locked lake layout (``docs/adr/0001-shared-artifact-contract.md``)::

    {league}/{level}/{run_date}/<relative-file>
    {league}/{level}/latest/<relative-file>
    {league}/{level}/latest/manifest.json

``<relative-file>`` is any path under local ``artifacts/`` (flat CSVs today;
nested files are first-class). ``run_date`` is ``YYYY-MM-DD`` (UTC unless
``ARTIFACTS_RUN_DATE`` is set). ``latest/`` is a copy of the most recent
successful publish so dashboards do not need to list history. Future
minor-league feeds use the same shape (e.g. ``milb/aaa/2026-08-23/``).
Fantasy cards: ``fantasy/cards.jsonl`` (ranked #111 emitter).
See ``docs/adr/0001-shared-artifact-contract.md``.

A brief read-only compat bridge still accepts the #109
``{league}/{level}/latest/`` prefix so already-published objects keep working.

URI schemes: ``s3://``, ``r2://``, ``gs://``, ``file://`` (and bare paths).
Backends implement a small put/get protocol (fsspec-style, no vendor lock).
``file://`` works without boto3 and is the CI / QA path.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Literal, Protocol
from urllib.parse import urlparse

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.fantasy import (
    FANTASY_CARDS_RELPATH,
    FANTASY_SCHEMA_VERSION,
    emit_ranked_fantasy_cards,
)

log = logging.getLogger(__name__)

CurrentPromoteDecision = Literal["promote", "skip_soft", "fail_closed"]

SCHEMA_VERSION = FANTASY_SCHEMA_VERSION
RUNS_PREFIX = "runs"
CURRENT_PREFIX = "current"
LEGACY_LATEST = "latest"
MANIFEST_NAME = "manifest.json"
SKIP_DIR_NAMES = {".remote_cache", ".cache", "__pycache__", ".current_staging", ".current_prev"}
PLOT_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".html", ".pdf", ".pkl", ".joblib"}
RESERVED_RUN_IDS = frozenset({CURRENT_PREFIX, LEGACY_LATEST, RUNS_PREFIX})
REQUIRED_MANIFEST_FIELDS = (
    "schema_version",
    "as_of_date",
    "created_at",
    "git_sha",
    "pipeline_steps",
    "war_source_summary",
    "files",
)
DEFAULT_PIPELINE_STEPS = (
    "pull_sources",
    "pull_war",
    "pull_mlb_stats",
    "pull_sportsdataio",
    "build_warehouse",
    "build_metrics",
    "train_win_model",
    "cluster_teams",
)

_RUN_ID_TIMESTAMP = re.compile(r"^\d{8}T\d{6}Z$")
_RUN_ID_GENERIC = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

SourceBadge = Literal["remote", "local", "missing"]


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
    """fsspec-style object interface. Implementations must not assume a vendor."""

    def put(self, relative_key: str, data: bytes) -> None: ...

    def get(self, relative_key: str) -> bytes | None: ...


@dataclass
class UploadResult:
    uri: str
    run_id: str
    as_of_date: str
    relative_prefix: str
    files: list[str]
    skipped: bool = False
    reason: str | None = None
    current_updated: bool = False

    @property
    def run_date(self) -> str:
        """Alias of ``as_of_date`` (brief compat with the #109 field name)."""
        return self.as_of_date


@dataclass(frozen=True)
class ArtifactHit:
    path: Path
    source: SourceBadge


def parse_artifacts_uri(uri: str) -> ParsedURI:
    """Parse an object-store or filesystem artifacts URI."""
    text = str(uri).strip()
    if not text:
        raise ValueError("artifacts URI is empty")
    parsed = urlparse(text)
    scheme = (parsed.scheme or "").lower()
    if scheme in {"s3", "s3a", "r2", "gs"}:
        bucket = parsed.netloc
        if not bucket:
            raise ValueError(f"{scheme} URI is missing a bucket: {uri!r}")
        prefix = parsed.path.lstrip("/").rstrip("/")
        # r2/gs share the same put/get backend as s3 (endpoint via env).
        normalized = "s3" if scheme in {"s3", "s3a", "r2"} else "gs"
        return ParsedURI(scheme=normalized, bucket=bucket, prefix=prefix, raw=text)
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
        "Use s3://, r2://, gs://, or file:///path "
        "(see docs/adr/0001-shared-artifact-contract.md)."
    )


def run_prefix(run_id: str) -> str:
    return f"{RUNS_PREFIX}/{_run_id_token(run_id)}"


def run_object_key(run_id: str, relpath: str) -> str:
    return f"{run_prefix(run_id)}/{_relpath(relpath)}"


def current_object_key(relpath: str) -> str:
    return f"{CURRENT_PREFIX}/{_relpath(relpath)}"


def partition_key(league: str, level: str, run_date: str) -> str:
    """Legacy #109 key ``{league}/{level}/{run_date}`` (compat bridge only)."""
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


def default_as_of_date(
    *,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    override = (
        env.get("ARTIFACTS_AS_OF_DATE", "").strip()
        or env.get("ARTIFACTS_RUN_DATE", "").strip()
    )
    if override:
        return _iso_date_token(override)
    current = now or datetime.now(timezone.utc)
    return current.date().isoformat()


def default_run_date(
    *,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """#109 name for ``default_as_of_date``."""
    return default_as_of_date(now=now, environ=environ)


def default_run_id(
    *,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    env = os.environ if environ is None else environ
    override = env.get("ARTIFACTS_RUN_ID", "").strip() or env.get("GITHUB_RUN_ID", "").strip()
    if override:
        return _run_id_token(override)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.strftime("%Y%m%dT%H%M%SZ")


def detect_git_sha(
    *,
    environ: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
) -> str | None:
    env = os.environ if environ is None else environ
    sha = (env.get("GITHUB_SHA") or env.get("ARTIFACTS_GIT_SHA") or "").strip()
    if sha:
        return sha
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    text = out.strip()
    return text or None


def classify_artifact_relpath(relpath: str) -> str:
    """Map a local relative path onto the locked lake tree."""
    rel = _relpath(relpath)
    name = Path(rel).name
    if name == MANIFEST_NAME:
        return MANIFEST_NAME
    if name == "metrics_manifest.json":
        return "metrics/metrics_manifest.json"
    if rel.startswith(("metrics/", "models/", "fantasy/")):
        if rel.startswith("fantasy/") and name != Path(FANTASY_CARDS_RELPATH).name:
            return FANTASY_CARDS_RELPATH
        return rel
    if name == Path(FANTASY_CARDS_RELPATH).name:
        return FANTASY_CARDS_RELPATH
    suffix = Path(name).suffix.lower()
    if suffix in PLOT_SUFFIXES:
        return f"models/{name}"
    if suffix == ".csv":
        return f"metrics/{name}"
    return f"models/{name}"


def artifact_source_label(
    settings: ArtifactSettings,
    *,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> SourceBadge:
    """Return the locked source badge: ``remote`` | ``local`` | ``missing``."""
    if settings.uri:
        try:
            store = backend if backend is not None else open_backend(settings.uri, environ=environ)
            if _remote_pointer_present(store, settings):
                return "remote"
        except Exception as exc:
            log.warning("Remote artifact probe failed (%s); checking local fallback", exc)
    if _local_artifacts_present(settings.local_dir):
        return "local"
    return "missing"


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
        if path.name == MANIFEST_NAME:
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
    """S3-compatible object store (AWS S3, Cloudflare R2, GCS XML, MinIO, …)."""

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
    """Build a boto3 S3 client from standard AWS / R2 / GCS-interop env vars."""
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
    if parsed.scheme in {"s3", "gs"}:
        client = s3_client if s3_client is not None else build_s3_client(environ)
        return S3Backend(parsed.bucket or "", parsed.prefix, client)
    return FileBackend(Path(parsed.prefix))


def build_manifest(
    *,
    run_id: str,
    as_of_date: str,
    files: Sequence[str],
    created_at: str,
    git_sha: str | None,
    pipeline_steps: Sequence[str] | None,
    war_source_summary: Mapping[str, int] | None,
    schema_version: str = SCHEMA_VERSION,
) -> dict[str, object]:
    manifest = {
        "schema_version": schema_version,
        "as_of_date": as_of_date,
        "created_at": created_at,
        "git_sha": git_sha,
        "run_id": run_id,
        "pipeline_steps": list(pipeline_steps or DEFAULT_PIPELINE_STEPS),
        "war_source_summary": dict(war_source_summary or {"bbref": 0, "approx": 0, "mixed": 0}),
        "files": list(files),
    }
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ArtifactUploadError(f"manifest.json missing required fields: {missing}")
    return manifest


def summarize_war_sources(local_dir: str | Path) -> dict[str, int]:
    """Count ``war_source`` on player metrics. Warehouse ``real`` → card ``bbref``."""
    summary = {"bbref": 0, "approx": 0, "mixed": 0}
    path = _find_player_metrics_csv(Path(local_dir))
    if path is None:
        return summary
    try:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                src = (row.get("war_source") or "approx").strip().lower()
                if src in {"real", "bbref"}:
                    summary["bbref"] += 1
                elif src == "mixed":
                    summary["mixed"] += 1
                else:
                    summary["approx"] += 1
    except OSError:
        return summary
    return summary


def upload_artifacts(
    local_dir: str | Path,
    settings: ArtifactSettings,
    *,
    run_id: str | None = None,
    run_date: str | None = None,
    as_of_date: str | None = None,
    backend: ArtifactBackend | None = None,
    now: datetime | None = None,
    environ: Mapping[str, str] | None = None,
    pipeline_steps: Sequence[str] | None = None,
    git_sha: str | None = None,
    update_current: bool = True,
) -> UploadResult:
    """Publish local artifacts to immutable ``runs/{run_id}/``, then ``current/``."""
    env = os.environ if environ is None else environ
    stamp = now or datetime.now(timezone.utc)
    resolved_as_of = as_of_date or run_date or default_as_of_date(now=stamp, environ=env)
    resolved_run = run_id or default_run_id(now=stamp, environ=env)

    if not settings.uri:
        return UploadResult(
            uri="",
            run_id=resolved_run,
            as_of_date=resolved_as_of,
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
    _ensure_fantasy_cards(source, resolved_as_of)
    files = iter_artifact_files(source)

    store = backend if backend is not None else open_backend(settings.uri, environ=env)
    classified: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path in files:
        rel = classify_artifact_relpath(path.relative_to(source).as_posix())
        if rel == MANIFEST_NAME or rel in seen:
            continue
        seen.add(rel)
        classified.append((path, rel))
    relative_names = [rel for _, rel in classified]
    if FANTASY_CARDS_RELPATH not in seen:
        raise ArtifactUploadError(
            f"Refusing to publish without {FANTASY_CARDS_RELPATH}"
        )

    created = stamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if stamp.tzinfo else stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = build_manifest(
        run_id=resolved_run,
        as_of_date=resolved_as_of,
        files=relative_names,
        created_at=created,
        git_sha=git_sha if git_sha is not None else detect_git_sha(environ=env),
        pipeline_steps=pipeline_steps,
        war_source_summary=summarize_war_sources(source),
    )
    payload = json.dumps(manifest, indent=2).encode("utf-8")
    run_root = run_prefix(resolved_run)

    try:
        existing = store.get(run_object_key(resolved_run, MANIFEST_NAME))
        if existing is not None:
            raise ArtifactUploadError(
                f"Refusing to mutate immutable run {resolved_run!r} under {run_root}/"
            )
        for path, rel in classified:
            store.put(run_object_key(resolved_run, rel), path.read_bytes())
        store.put(run_object_key(resolved_run, MANIFEST_NAME), payload)
    except ArtifactStoreError:
        raise
    except Exception as exc:
        raise ArtifactUploadError(
            f"Failed to upload artifacts to {settings.uri}: {exc}"
        ) from exc

    current_updated = False
    promote_reason: str | None = None
    if update_current:
        decision, promote_reason = evaluate_current_promote(source)
        if decision == "fail_closed":
            raise ArtifactUploadError(
                f"Run {resolved_run} is stored but current/ promote refused: {promote_reason}"
            )
        if decision == "skip_soft":
            log.warning(
                "Skipping current/ promote for run %s (%s); leaving prior current/ in place",
                resolved_run,
                promote_reason,
            )
        else:
            try:
                _promote_current(store, resolved_run, relative_names, payload)
                current_updated = True
            except ArtifactStoreError:
                raise
            except Exception as exc:
                raise ArtifactUploadError(
                    f"Run {resolved_run} is stored but current/ promote failed: {exc}"
                ) from exc

    log.info(
        "Uploaded %d artifact files to %s/%s%s",
        len(relative_names),
        settings.uri.rstrip("/"),
        run_root,
        f" and {CURRENT_PREFIX}/" if current_updated else "",
    )
    return UploadResult(
        uri=settings.uri,
        run_id=resolved_run,
        as_of_date=resolved_as_of,
        relative_prefix=run_root,
        files=relative_names,
        current_updated=current_updated,
        reason=None if current_updated else promote_reason,
    )


def resolve_artifact(
    filename: str,
    settings: ArtifactSettings | None = None,
    *,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Return a local path for ``filename``, preferring shared ``current/``.

    Order: fresh remote cache → remote ``current/`` (compat ``latest/``) →
    stale remote cache → local ``artifacts_dir`` → ``None``.
    Remote errors are logged and treated as a miss so the dashboard still
    loads from disk when the store is unreachable.
    """
    hit = resolve_artifact_hit(filename, settings, backend=backend, environ=environ)
    return None if hit is None else hit.path


def resolve_artifact_hit(
    filename: str,
    settings: ArtifactSettings | None = None,
    *,
    backend: ArtifactBackend | None = None,
    environ: Mapping[str, str] | None = None,
) -> ArtifactHit | None:
    cfg = settings if settings is not None else load_artifact_settings(environ=environ)
    rel = _lookup_relpath(filename)
    if not rel:
        return None
    cache_path = cfg.cache_dir / CURRENT_PREFIX / rel
    local_path = _first_existing(_local_candidates(cfg.local_dir, rel))

    if cfg.uri:
        if _cache_is_fresh(cache_path, cfg.cache_ttl_s):
            return ArtifactHit(cache_path, "remote")
        try:
            store = backend if backend is not None else open_backend(cfg.uri, environ=environ)
            data, key = _get_first(store, remote_lookup_keys(rel, cfg))
        except Exception as exc:
            log.warning(
                "Remote artifact %s unavailable (%s); falling back to local",
                rel,
                exc,
            )
        else:
            if data is not None:
                dest = cfg.cache_dir / key
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                return ArtifactHit(dest, "remote")
        if cache_path.is_file():
            return ArtifactHit(cache_path, "remote")

    if local_path is not None:
        return ArtifactHit(local_path, "local")
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


def remote_lookup_keys(relpath: str, settings: ArtifactSettings) -> list[str]:
    """Candidate remote keys: ``current/`` first, then the #109 latest/ bridge."""
    rel = _lookup_relpath(relpath)
    name = Path(rel).name
    rels = [rel]
    if not rel.startswith("metrics/") and name.endswith(".csv"):
        rels.append(f"metrics/{name}")
    if not rel.startswith("models/"):
        rels.append(f"models/{name}")
    if name == Path(FANTASY_CARDS_RELPATH).name:
        rels.append(FANTASY_CARDS_RELPATH)
    rels.append(name)
    keys: list[str] = []
    for item in rels:
        keys.append(current_object_key(item))
    for item in rels:
        keys.append(f"{LEGACY_LATEST}/{item}")
        keys.append(object_key(settings.league, settings.level, LEGACY_LATEST, item))
        keys.append(object_key(settings.league, settings.level, LEGACY_LATEST, name))
    return list(dict.fromkeys(keys))


def publish_nightly_artifacts(
    config_path: str = "config/settings.yaml",
    *,
    settings: ArtifactSettings | None = None,
    backend: ArtifactBackend | None = None,
    run_id: str | None = None,
    run_date: str | None = None,
    as_of_date: str | None = None,
    environ: Mapping[str, str] | None = None,
    pipeline_steps: Sequence[str] | None = None,
    git_sha: str | None = None,
) -> UploadResult:
    """Upload ``artifacts_dir`` after a successful nightly pipeline run."""
    cfg = settings if settings is not None else load_artifact_settings(config_path, environ=environ)
    if not cfg.uri:
        log.info("ARTIFACTS_URI unset; skipping shared-storage upload")
        return UploadResult(
            uri="",
            run_id=run_id or default_run_id(environ=environ),
            as_of_date=as_of_date or run_date or default_as_of_date(environ=environ),
            relative_prefix="",
            files=[],
            skipped=True,
            reason="no_uri",
        )
    return upload_artifacts(
        cfg.local_dir,
        cfg,
        run_id=run_id,
        run_date=run_date,
        as_of_date=as_of_date,
        backend=backend,
        environ=environ,
        pipeline_steps=pipeline_steps,
        git_sha=git_sha,
        update_current=True,
    )


def _ensure_fantasy_cards(local_dir: Path, as_of_date: str) -> Path:
    """Emit ranked cards from published metrics when the lake file is missing."""
    dest = local_dir / FANTASY_CARDS_RELPATH
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    return emit_ranked_fantasy_cards(local_dir, as_of_date=as_of_date)


def decide_current_promote(
    *,
    sdio_in_season: bool | None = None,
    active_season: int | None = None,
    metrics_max_season: int | None = None,
    current_season_missing: bool | None = None,
) -> CurrentPromoteDecision:
    """Decide whether ``current/`` may be swapped onto this run.

    Fail-closed when SportsDataIO landed in-season data but published
    metrics have ``max(season) < Y``. Missing-key / empty SDIO skips
    promote without failing so a prior ``current/`` stays put.
    """
    if sdio_in_season and active_season is not None:
        if metrics_max_season is None or int(metrics_max_season) < int(active_season):
            return "fail_closed"
    if current_season_missing and not sdio_in_season:
        return "skip_soft"
    return "promote"


def evaluate_current_promote(local_dir: str | Path) -> tuple[CurrentPromoteDecision, str]:
    """Read metrics coverage from ``local_dir`` and return (decision, reason)."""
    coverage = _read_metrics_coverage(Path(local_dir))
    if coverage is None:
        return "promote", ""
    active = _optional_int(coverage.get("active_season"))
    max_season = _coverage_max_season(coverage, Path(local_dir))
    decision = decide_current_promote(
        sdio_in_season=coverage.get("sdio_in_season"),
        active_season=active,
        metrics_max_season=max_season,
        current_season_missing=coverage.get("current_season_missing"),
    )
    if decision == "fail_closed":
        return (
            decision,
            f"SDIO in-season data present but metrics max(season)={max_season} "
            f"< active year {active}",
        )
    if decision == "skip_soft":
        reason = coverage.get("current_season_missing_reason") or "current_season_missing"
        return decision, str(reason)
    return decision, ""


def _read_metrics_coverage(local_dir: Path) -> dict[str, object] | None:
    for path in (
        local_dir / "metrics_manifest.json",
        local_dir / "metrics" / "metrics_manifest.json",
    ):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _coverage_max_season(coverage: Mapping[str, object], local_dir: Path) -> int | None:
    present = coverage.get("seasons_present")
    if isinstance(present, (list, tuple)) and present:
        years = [_optional_int(year) for year in present]
        found = [year for year in years if year is not None]
        if found:
            return max(found)
    return _metrics_csv_max_season(local_dir)


def _metrics_csv_max_season(local_dir: Path) -> int | None:
    path = _find_player_metrics_csv(local_dir)
    if path is None:
        return None
    try:
        import csv

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            years: list[int] = []
            for row in reader:
                raw = row.get("season") or row.get("year_id") or row.get("season_key")
                year = _optional_int(raw)
                if year is not None:
                    years.append(year)
    except OSError:
        return None
    return max(years) if years else None


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _promote_current(
    store: ArtifactBackend,
    run_id: str,
    files: Sequence[str],
    manifest: bytes,
) -> None:
    """Copy a finished run onto ``current/``. Never writes back into ``runs/``."""
    if isinstance(store, FileBackend):
        _promote_current_filesystem(store.root, run_id)
        return
    for rel in files:
        data = store.get(run_object_key(run_id, rel))
        if data is None:
            raise ArtifactUploadError(f"Run object missing during current/ promote: {rel}")
        store.put(current_object_key(rel), data)
    store.put(current_object_key(MANIFEST_NAME), manifest)


def _promote_current_filesystem(root: Path, run_id: str) -> None:
    src = root / RUNS_PREFIX / run_id
    if not src.is_dir():
        raise ArtifactUploadError(f"Run directory missing; cannot promote current/: {src}")
    staging = root / ".current_staging"
    previous = root / ".current_prev"
    dest = root / CURRENT_PREFIX
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(src, staging)
    if dest.exists():
        if previous.exists():
            shutil.rmtree(previous)
        dest.rename(previous)
    staging.rename(dest)
    if previous.exists():
        shutil.rmtree(previous)


def _remote_pointer_present(store: ArtifactBackend, settings: ArtifactSettings) -> bool:
    if store.get(current_object_key(MANIFEST_NAME)) is not None:
        return True
    if store.get(f"{LEGACY_LATEST}/{MANIFEST_NAME}") is not None:
        return True
    legacy = object_key(settings.league, settings.level, LEGACY_LATEST, MANIFEST_NAME)
    return store.get(legacy) is not None


def _local_artifacts_present(local_dir: Path) -> bool:
    return any(iter_artifact_files(local_dir)) or (local_dir / FANTASY_CARDS_RELPATH).is_file()


def _lookup_relpath(filename: str) -> str:
    rel = str(filename).replace("\\", "/").lstrip("/")
    if not rel or rel.endswith("/"):
        return ""
    name = Path(rel).name
    if name == Path(FANTASY_CARDS_RELPATH).name or rel.endswith(FANTASY_CARDS_RELPATH):
        return FANTASY_CARDS_RELPATH
    if rel.startswith(("metrics/", "models/", "fantasy/", f"{CURRENT_PREFIX}/")):
        return rel[len(f"{CURRENT_PREFIX}/") :] if rel.startswith(f"{CURRENT_PREFIX}/") else rel
    return classify_artifact_relpath(name)


def _local_candidates(local_dir: Path, relpath: str) -> list[Path]:
    rel = _lookup_relpath(relpath)
    name = Path(rel).name
    return [
        local_dir / rel,
        local_dir / name,
        local_dir / "metrics" / name,
        local_dir / "models" / name,
        local_dir / "fantasy" / name,
        local_dir / CURRENT_PREFIX / rel,
    ]


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def _get_first(store: ArtifactBackend, keys: Sequence[str]) -> tuple[bytes | None, str]:
    for key in keys:
        data = store.get(key)
        if data is not None:
            return data, key
    return None, ""


def _find_player_metrics_csv(local_dir: Path) -> Path | None:
    return _first_existing(
        [
            local_dir / "metrics" / "player_season_metrics.csv",
            local_dir / "player_season_metrics.csv",
        ]
    )


def _relpath(filename: str) -> str:
    rel = str(filename).replace("\\", "/").lstrip("/")
    if not rel or rel.endswith("/") or ".." in Path(rel).parts:
        raise ValueError(f"Invalid artifact filename: {filename!r}")
    return rel


def _run_id_token(value: str) -> str:
    text = str(value).strip()
    if not text or text in RESERVED_RUN_IDS or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid run_id: {value!r}")
    if _RUN_ID_TIMESTAMP.match(text) or _RUN_ID_GENERIC.match(text):
        return text
    raise ValueError(
        f"Invalid run_id {value!r}; expected YYYYMMDDTHHMMSSZ or a GitHub Actions run id"
    )


def _iso_date_token(value: str) -> str:
    text = str(value).strip()
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _partition_token(value: str, name: str, *, allow_latest: bool = False) -> str:
    text = str(value).strip()
    if name in {"league", "level"}:
        text = text.lower()
    if not text or "/" in text or "\\" in text or text in {".", ".."}:
        raise ValueError(f"Invalid {name}: {value!r}")
    if name == "run_date" and text != LEGACY_LATEST:
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"Invalid run_date {value!r}; expected YYYY-MM-DD") from exc
    if name == "run_date" and text == LEGACY_LATEST and not allow_latest:
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
