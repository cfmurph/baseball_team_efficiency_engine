"""
Pull MLB Stats API majors feeds into versioned raw.

Public API, no key. Soft-fails on network/API blips so nightly can continue
on the Lahman + BR rWAR path. Does not replace Baseball-Reference rWAR.

Usage
-----
    python3 -m pipeline.extract.pull_mlb_stats
    python3 -m pipeline.extract.pull_mlb_stats --season 2024 --season 2025
    ARTIFACTS_URI=file:///tmp/btee-qa python3 -m pipeline.extract.pull_mlb_stats
"""
from __future__ import annotations

import logging

import typer

from src.baseball_analytics.config import load_artifact_settings, load_settings
from src.baseball_analytics.io import ensure_dir
from src.baseball_analytics.mlb_stats import (
    client_from_settings,
    open_optional_backend,
    pull_majors_feeds,
    seasons_from_settings,
    write_raw_payload,
    ENDPOINT_EXTRACT_REPORT,
)
from src.baseball_analytics.storage import default_as_of_date as storage_as_of_date

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Pull MLB Stats API majors feeds (soft-fail).")


@app.command()
def main(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings YAML"),
    as_of_date: str | None = typer.Option(None, help="Partition date YYYY-MM-DD"),
    season: list[int] | None = typer.Option(None, "--season", help="Season year(s) to pull"),
    schedule_mode: str = typer.Option("season", help="schedule: season (default) or date"),
) -> None:
    """Land majors team/player/game JSON under raw/mlb_stats. Always exits 0 on API errors."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    artifact_settings = load_artifact_settings(config_path)
    raw_dir = ensure_dir(settings["raw_dir"])
    resolved_date = as_of_date or storage_as_of_date()
    seasons = list(season) if season else seasons_from_settings(settings, resolved_date)
    backend = open_optional_backend(artifact_settings.uri)
    client = client_from_settings(settings)

    log.info(
        "MLB Stats API extract as_of=%s seasons=%s uri=%s",
        resolved_date,
        seasons,
        artifact_settings.uri or "(local data/raw/mlb_stats)",
    )
    try:
        report = pull_majors_feeds(
            raw_dir=raw_dir,
            as_of_date=resolved_date,
            seasons=seasons,
            client=client,
            backend=backend,
            schedule_mode=schedule_mode,
        )
    except Exception as exc:
        log.warning("Stats API extract failed softly: %s", exc)
        failed = {
            "as_of_date": resolved_date,
            "seasons": seasons,
            "soft_fail": True,
            "ok": False,
            "error": str(exc),
            "endpoints": [],
        }
        try:
            write_raw_payload(
                failed,
                endpoint=ENDPOINT_EXTRACT_REPORT,
                as_of_date=resolved_date,
                filename="extract_report.json",
                raw_dir=raw_dir,
                backend=backend,
            )
        except Exception as write_exc:
            log.warning("Could not write extract report: %s", write_exc)
        typer.echo(f"MLB Stats API extract soft-failed; warehouse will use Lahman-only. ({exc})")
        raise typer.Exit(code=0) from None

    failed_eps = [item.endpoint for item in report.endpoints if not item.ok]
    landed = [item.relative_key for item in report.endpoints if item.ok]
    if failed_eps:
        log.warning("Soft-failed endpoints: %s", ", ".join(failed_eps))
    log.info("Landed %d raw objects under mlb_stats/%s", len(landed), resolved_date)
    typer.echo(
        f"MLB Stats API extract complete ({len(landed)} ok, {len(failed_eps)} soft-failed) "
        f"as_of={resolved_date}"
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
