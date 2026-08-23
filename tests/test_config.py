from __future__ import annotations

from src.baseball_analytics.config import load_artifact_settings, load_settings, resolve_artifacts_uri


def test_load_settings() -> None:
    settings = load_settings("config/settings.yaml")
    assert settings["min_year"] == 1990
    assert "teams" in settings["sources"]
    assert "batting" in settings["war_sources"]
    assert "pitching" in settings["war_sources"]
    assert settings["war_sources"]["batting"].endswith("war_daily_bat.txt")
    assert settings["artifacts_partition"]["league"] == "mlb"
    assert settings["artifacts_partition"]["level"] == "mlb"
    assert settings["mlb_stats"]["base_url"] == "https://statsapi.mlb.com"
    assert settings["mlb_stats"]["team_crosswalk"] == "data/crosswalks/mlb_team_map.csv"
    assert settings["sportsdataio"]["base_url"] == "https://api.sportsdata.io"
    assert settings["sportsdataio"]["team_crosswalk"] == "data/crosswalks/mlb_team_map.csv"


def test_resolve_artifacts_uri_precedence() -> None:
    settings = {"artifacts_uri": "s3://yaml-bucket/prefix"}
    assert resolve_artifacts_uri(settings, environ={}) == "s3://yaml-bucket/prefix"
    assert (
        resolve_artifacts_uri(settings, environ={"ARTIFACTS_URI": "s3://env-bucket/data"})
        == "s3://env-bucket/data"
    )
    assert resolve_artifacts_uri({"artifacts_uri": ""}, environ={}) is None


def test_load_artifact_settings_defaults_when_yaml_missing(tmp_path) -> None:
    cfg = load_artifact_settings(str(tmp_path / "missing.yaml"), environ={})
    assert cfg.uri is None
    assert cfg.league == "mlb"
    assert cfg.level == "mlb"
    assert cfg.local_dir.name == "artifacts"
