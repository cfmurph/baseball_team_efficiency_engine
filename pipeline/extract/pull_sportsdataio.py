"""
Pull SportsDataIO majors feeds into versioned raw (Phase 0 / schema v0.1).

Reads SPORTSDATAIO_API_KEY from the environment. Soft-fails (exit 0) when
the key is missing or an endpoint blips so nightly / CI still pass.
Does not replace Lahman, BR rWAR, or the MLB Stats API path.

Usage
-----
    python3 -m pipeline.extract.pull_sportsdataio
    python3 -m pipeline.extract.pull_sportsdataio --season 2024 --as-of-date 2026-08-23
    ARTIFACTS_URI=file:///tmp/btee-qa python3 -m pipeline.extract.pull_sportsdataio
"""
from __future__ import annotations

import logging

import typer

from src.baseball_analytics.config import load_artifact_settings, load_settings
from src.baseball_analytics.io import ensure_dir
from src.baseball_analytics.sportsdataio import (
    API_KEY_ENV,
    ENDPOINT_EXTRACT_REPORT,
    NIGHTLY_EXTRACT_REPORT_NAME,
    ExtractReport,
    client_from_settings,
    mark_extract_season_coverage,
    open_optional_backend,
    pull_phase0_feeds,
    record_skipped_window_statuses,
    resolve_api_key,
    seasons_from_settings,
    stage_extract_report,
    write_raw_payload,
)
from src.baseball_analytics.storage import default_as_of_date as storage_as_of_date

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Pull SportsDataIO feeds (soft-fail without key).")


@app.command()
def main(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings YAML"),
    as_of_date: str | None = typer.Option(None, help="Partition date YYYY-MM-DD"),
    season: list[int] | None = typer.Option(None, "--season", help="Season year(s) for season stub"),
    include_season_feeds: bool = typer.Option(
        False,
        "--include-season-feeds/--no-include-season-feeds",
        help="Also pull season-wide Games/{season} (default: incremental date feeds)",
    ),
) -> None:
    """Land SDIO JSON under raw/sportsdataio. Always exits 0 on missing key / API errors."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    artifact_settings = load_artifact_settings(config_path)
    raw_dir = ensure_dir(settings["raw_dir"])
    artifacts_dir = ensure_dir(settings.get("artifacts_dir") or artifact_settings.local_dir)
    resolved_date = as_of_date or storage_as_of_date()
    seasons = list(season) if season else seasons_from_settings(settings, resolved_date)
    backend = open_optional_backend(artifact_settings.uri)
    api_key = resolve_api_key()
    client = client_from_settings(settings, api_key=api_key)

    log.info(
        "SportsDataIO extract as_of=%s seasons=%s key=%s uri=%s",
        resolved_date,
        seasons,
        "set" if api_key else "missing",
        artifact_settings.uri or "(local data/raw/sportsdataio)",
    )
    try:
        report = pull_phase0_feeds(
            raw_dir=raw_dir,
            as_of_date=resolved_date,
            seasons=seasons,
            client=client,
            backend=backend,
            include_season_feeds=include_season_feeds,
            artifacts_dir=artifacts_dir,
        )
    except Exception as exc:
        log.warning("SportsDataIO extract failed softly: %s", exc)
        failed_report = ExtractReport(
            as_of_date=resolved_date,
            seasons=seasons,
            soft_fail=True,
            ok=False,
            error=str(exc),
            skipped_reason="extract_exception",
        )
        record_skipped_window_statuses(failed_report)
        mark_extract_season_coverage(failed_report)
        failed = failed_report.to_dict()
        try:
            write_raw_payload(
                failed,
                endpoint=ENDPOINT_EXTRACT_REPORT,
                as_of_date=resolved_date,
                filename=NIGHTLY_EXTRACT_REPORT_NAME,
                raw_dir=raw_dir,
                backend=backend,
            )
            stage_extract_report(failed, artifacts_dir)
        except Exception as write_exc:
            log.warning("Could not write extract report: %s", write_exc)
        typer.echo(
            f"SportsDataIO extract soft-failed; warehouse will skip the spine. ({exc})"
        )
        raise typer.Exit(code=0) from None

    failed_eps = [item.endpoint for item in report.endpoints if not item.ok]
    landed = [item.relative_key for item in report.endpoints if item.ok]
    if report.skipped_reason == "missing_api_key":
        typer.echo(
            f"SportsDataIO extract skipped ({API_KEY_ENV} unset); "
            f"warehouse will skip the spine. as_of={resolved_date}"
        )
        raise typer.Exit(code=0)
    if failed_eps:
        log.warning("Soft-failed endpoints: %s", ", ".join(failed_eps))
    log.info("Landed %d raw objects under sportsdataio/%s", len(landed), resolved_date)
    typer.echo(
        f"SportsDataIO extract complete ({len(landed)} ok, {len(failed_eps)} soft-failed) "
        f"as_of={resolved_date}"
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
