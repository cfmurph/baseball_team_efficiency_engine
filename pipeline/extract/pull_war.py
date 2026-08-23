"""
Download Baseball-Reference daily WAR files.

Lahman extract (pull_sources.py) is unchanged.  This step is optional for the
warehouse to build, but without it every player-season falls back to the
wOBA/FIP approximation.

Usage
-----
    python3 -m pipeline.extract.pull_war
"""
from __future__ import annotations

from pathlib import Path
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import download_file, ensure_dir
from src.baseball_analytics.war import BR_BAT_FILENAME, BR_PIT_FILENAME

app = typer.Typer(add_completion=False)

_FILENAME = {
    "batting": BR_BAT_FILENAME,
    "pitching": BR_PIT_FILENAME,
}


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    raw_dir = ensure_dir(settings["raw_dir"])
    war_sources = settings.get("war_sources") or {}

    downloaded = 0
    for name, url in war_sources.items():
        if not isinstance(url, str) or not url.startswith("http"):
            continue
        filename = _FILENAME.get(name, f"{name}.txt")
        output_path = Path(raw_dir) / filename
        download_file(url, output_path, timeout=180)
        typer.echo(f"Downloaded {name} WAR -> {output_path}")
        downloaded += 1

    if downloaded == 0:
        raise typer.BadParameter(
            "No http(s) URLs in settings['war_sources']. "
            "Expected batting/pitching Baseball-Reference war_daily_*.txt URLs."
        )


if __name__ == "__main__":
    app()
