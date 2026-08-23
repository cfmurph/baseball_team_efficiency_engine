"""Offline contracts for Baseball-Reference rWAR extract (no live download)."""
from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from pipeline.extract import pull_war
from src.baseball_analytics.config import load_settings
from src.baseball_analytics.war import BR_BAT_FILENAME, BR_PIT_FILENAME

BR_BAT_URL = "https://www.baseball-reference.com/data/war_daily_bat.txt"
BR_PIT_URL = "https://www.baseball-reference.com/data/war_daily_pitch.txt"


def test_settings_war_sources_point_at_br_http_files() -> None:
    settings = load_settings("config/settings.yaml")
    war = settings["war_sources"]
    assert war["batting"] == BR_BAT_URL
    assert war["pitching"] == BR_PIT_URL
    assert war["batting"].startswith("https://")
    assert war["pitching"].startswith("https://")
    assert Path(war["team_map"]).as_posix() == "data/crosswalks/br_team_map.csv"


def test_pull_war_filename_map_matches_war_constants() -> None:
    assert pull_war._FILENAME["batting"] == BR_BAT_FILENAME == "war_daily_bat.txt"
    assert pull_war._FILENAME["pitching"] == BR_PIT_FILENAME == "war_daily_pitch.txt"


def test_pull_war_downloads_only_http_sources(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(url, output_path, timeout=180):
        calls.append((url, Path(output_path)))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("stub\n", encoding="utf-8")
        return Path(output_path)

    monkeypatch.setattr(pull_war, "download_file", fake_download)

    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tmp_path / "raw"),
                "war_sources": {
                    "batting": BR_BAT_URL,
                    "pitching": BR_PIT_URL,
                    "team_map": "data/crosswalks/br_team_map.csv",
                },
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(pull_war.app, ["--config-path", str(cfg)])
    assert result.exit_code == 0, result.output
    assert {url for url, _ in calls} == {BR_BAT_URL, BR_PIT_URL}
    assert {path.name for _, path in calls} == {BR_BAT_FILENAME, BR_PIT_FILENAME}


def test_pull_war_errors_when_no_http_urls(tmp_path) -> None:
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "raw_dir": str(tmp_path / "raw"),
                "war_sources": {"team_map": "data/crosswalks/br_team_map.csv"},
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(pull_war.app, ["--config-path", str(cfg)])
    assert result.exit_code != 0
    assert "war_sources" in result.output
