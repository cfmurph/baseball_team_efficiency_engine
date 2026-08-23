"""Run the full nightly data-refresh chain.

Each step is the same ``python3 -m …`` module used for manual runs.
The chain stops on the first non-zero exit so a broken step is obvious.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import typer

from src.baseball_analytics.storage import (
    ArtifactUploadError,
    publish_nightly_artifacts,
)

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parents[1]

# (display name, module) — keep in lockstep with README / AGENTS.md
PIPELINE_STEPS: tuple[tuple[str, str], ...] = (
    ("pull_sources", "pipeline.extract.pull_sources"),
    ("pull_war", "pipeline.extract.pull_war"),
    ("build_warehouse", "pipeline.transform.build_warehouse"),
    ("build_metrics", "pipeline.transform.build_metrics"),
    ("train_win_model", "models.train_win_model"),
    ("cluster_teams", "models.cluster_teams"),
)


@dataclass(frozen=True)
class StepResult:
    name: str
    module: str
    returncode: int
    duration_s: float


class PipelineStepError(RuntimeError):
    """Raised when a refresh step exits non-zero. Later steps are not run."""

    def __init__(
        self,
        *,
        name: str,
        module: str,
        returncode: int,
        remaining: list[str],
    ) -> None:
        self.name = name
        self.module = module
        self.returncode = returncode
        self.remaining = remaining
        leftover = ", ".join(remaining) if remaining else "(none)"
        super().__init__(
            f"Step '{name}' ({module}) failed with exit code {returncode}. "
            f"Not run: {leftover}."
        )


def _step_command(module: str, config_path: str, python_executable: str) -> list[str]:
    return [python_executable, "-m", module, "--config-path", config_path]


def run_pipeline(
    config_path: str = "config/settings.yaml",
    *,
    python_executable: str | None = None,
    cwd: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess] | None = None,
) -> list[StepResult]:
    """Execute every pipeline step in order.

    ``runner`` is ``subprocess.run`` by default and is injectable for tests.
    """
    python = python_executable or sys.executable
    workdir = Path(cwd) if cwd is not None else REPO_ROOT
    run = runner or subprocess.run

    results: list[StepResult] = []
    remaining = [name for name, _ in PIPELINE_STEPS]

    for index, (name, module) in enumerate(PIPELINE_STEPS, start=1):
        remaining.pop(0)
        cmd = _step_command(module, config_path, python)
        log.info("[%d/%d] Starting %s: %s", index, len(PIPELINE_STEPS), name, " ".join(cmd))
        started = time.monotonic()
        completed = run(cmd, cwd=str(workdir), check=False)
        duration = time.monotonic() - started
        returncode = int(getattr(completed, "returncode", 1))
        results.append(
            StepResult(name=name, module=module, returncode=returncode, duration_s=duration)
        )
        if returncode != 0:
            raise PipelineStepError(
                name=name,
                module=module,
                returncode=returncode,
                remaining=list(remaining),
            )
        log.info("[%d/%d] Finished %s in %.1fs", index, len(PIPELINE_STEPS), name, duration)

    return results


def refresh_and_publish(
    config_path: str = "config/settings.yaml",
    *,
    pipeline: Callable[..., list[StepResult]] | None = None,
    publish: Callable[..., object] | None = None,
    **pipeline_kwargs,
) -> list[StepResult]:
    """Run the pipeline, then upload artifacts when shared storage is configured."""
    run = pipeline or run_pipeline
    results = run(config_path, **pipeline_kwargs)
    publisher = publish or publish_nightly_artifacts
    try:
        publisher(
            config_path,
            pipeline_steps=[step.name for step in results],
        )
    except TypeError:
        # Tests inject ``publish=lambda path: …``.
        publisher(config_path)
    return results


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    """Refresh raw data, warehouse, metrics, and models."""
    _configure_logging()
    try:
        results = refresh_and_publish(config_path)
    except PipelineStepError as exc:
        log.error("%s", exc)
        raise typer.Exit(code=exc.returncode or 1) from exc
    except ArtifactUploadError as exc:
        log.error("Pipeline succeeded but artifact upload failed: %s", exc)
        raise typer.Exit(code=1) from exc

    total = sum(step.duration_s for step in results)
    for step in results:
        log.info("  %-18s %.1fs", step.name, step.duration_s)
    log.info("Nightly refresh complete in %.1fs", total)
    typer.echo("Nightly refresh complete")


if __name__ == "__main__":
    app()
