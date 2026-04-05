from __future__ import annotations

from pathlib import Path
import typer

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import download_csv, ensure_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    raw_dir = ensure_dir(settings["raw_dir"])

    for name, url in settings["sources"].items():
        output_path = Path(raw_dir) / f"{name}.csv"
        download_csv(url, output_path)
        typer.echo(f"Downloaded {name} -> {output_path}")


if __name__ == "__main__":
    app()
