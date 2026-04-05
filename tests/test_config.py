from __future__ import annotations

from src.baseball_analytics.config import load_settings


def test_load_settings() -> None:
    settings = load_settings("config/settings.yaml")
    assert settings["min_year"] == 1990
    assert "teams" in settings["sources"]
