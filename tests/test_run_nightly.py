"""Tests for the nightly refresh orchestrator."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.run_nightly import (
    PIPELINE_STEPS,
    PipelineStepError,
    _step_command,
    refresh_and_publish,
    run_pipeline,
)

pytestmark = pytest.mark.integration


def test_pipeline_steps_match_documented_chain() -> None:
    assert [module for _, module in PIPELINE_STEPS] == [
        "pipeline.extract.pull_sources",
        "pipeline.extract.pull_war",
        "pipeline.extract.pull_mlb_stats",
        "pipeline.extract.pull_sportsdataio",
        "pipeline.transform.build_warehouse",
        "pipeline.transform.build_metrics",
        "models.train_win_model",
        "models.cluster_teams",
    ]

def test_pull_war_follows_pull_sources_in_nightly_steps() -> None:
    """#110 contract: nightly refresh must download rWAR after Lahman extract."""
    names = [name for name, _ in PIPELINE_STEPS]
    assert "pull_war" in names, "PIPELINE_STEPS omitted pull_war (would rebuild on approx WAR)"
    assert names.index("pull_war") == names.index("pull_sources") + 1
    assert dict(PIPELINE_STEPS)["pull_war"] == "pipeline.extract.pull_war"

def test_pull_mlb_stats_follows_pull_war_in_nightly_steps() -> None:
    """#108: Stats API extract is after rWAR and soft-fails; warehouse stays Lahman-capable."""
    names = [name for name, _ in PIPELINE_STEPS]
    assert names.index("pull_mlb_stats") == names.index("pull_war") + 1
    assert dict(PIPELINE_STEPS)["pull_mlb_stats"] == "pipeline.extract.pull_mlb_stats"
    assert names.index("pull_sportsdataio") == names.index("pull_mlb_stats") + 1
    assert names.index("build_warehouse") == names.index("pull_sportsdataio") + 1


def test_pull_sportsdataio_follows_mlb_stats_in_nightly_steps() -> None:
    """#128: SDIO extract is after Stats API and soft-fails without the key."""
    names = [name for name, _ in PIPELINE_STEPS]
    assert dict(PIPELINE_STEPS)["pull_sportsdataio"] == "pipeline.extract.pull_sportsdataio"
    assert names.index("pull_sportsdataio") == names.index("pull_mlb_stats") + 1
    assert names.index("build_warehouse") == names.index("pull_sportsdataio") + 1

def test_step_command_forwards_config_path() -> None:
    cmd = _step_command("pipeline.extract.pull_sources", "config/settings.yaml", "/usr/bin/python3")
    assert cmd == [
        "/usr/bin/python3",
        "-m",
        "pipeline.extract.pull_sources",
        "--config-path",
        "config/settings.yaml",
    ]

def test_run_pipeline_executes_every_step_in_order(tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd, cwd, check):
        calls.append(cmd)
        assert check is False
        assert cwd == str(tmp_path)
        return SimpleNamespace(returncode=0)

    results = run_pipeline(
        "config/custom.yaml",
        python_executable="python3",
        cwd=tmp_path,
        runner=fake_runner,
    )

    assert [step.name for step in results] == [name for name, _ in PIPELINE_STEPS]
    assert all(step.returncode == 0 for step in results)
    assert [cmd[2] for cmd in calls] == [module for _, module in PIPELINE_STEPS]
    assert all(cmd[-1] == "config/custom.yaml" for cmd in calls)

def test_pull_war_failure_is_hard_and_skips_warehouse(tmp_path) -> None:
    """rWAR extract is fail-fast. Only pull_mlb_stats and pull_sportsdataio may soft-fail (exit 0)."""
    calls: list[str] = []

    def fake_runner(cmd, cwd, check):
        module = cmd[2]
        calls.append(module)
        returncode = 1 if module == "pipeline.extract.pull_war" else 0
        return SimpleNamespace(returncode=returncode)

    with pytest.raises(PipelineStepError) as exc_info:
        run_pipeline(
            python_executable="python3",
            cwd=tmp_path,
            runner=fake_runner,
        )

    err = exc_info.value
    assert err.name == "pull_war"
    assert err.returncode == 1
    assert "build_warehouse" in err.remaining
    assert "pull_mlb_stats" in err.remaining
    assert "pull_sportsdataio" in err.remaining
    assert calls == [
        "pipeline.extract.pull_sources",
        "pipeline.extract.pull_war",
    ]

def test_run_pipeline_stops_after_first_failure(tmp_path) -> None:
    calls: list[str] = []

    def fake_runner(cmd, cwd, check):
        module = cmd[2]
        calls.append(module)
        returncode = 0 if module != "pipeline.transform.build_metrics" else 7
        return SimpleNamespace(returncode=returncode)

    with pytest.raises(PipelineStepError) as exc_info:
        run_pipeline(
            python_executable="python3",
            cwd=tmp_path,
            runner=fake_runner,
        )

    err = exc_info.value
    assert err.name == "build_metrics"
    assert err.returncode == 7
    assert err.remaining == ["train_win_model", "cluster_teams"]
    assert calls == [
        "pipeline.extract.pull_sources",
        "pipeline.extract.pull_war",
        "pipeline.extract.pull_mlb_stats",
        "pipeline.extract.pull_sportsdataio",
        "pipeline.transform.build_warehouse",
        "pipeline.transform.build_metrics",
    ]
    assert "Not run: train_win_model, cluster_teams" in str(err)

def test_refresh_and_publish_uploads_only_after_success(tmp_path) -> None:
    published: list[str] = []

    def fake_pipeline(config_path, **kwargs):
        return []

    refresh_and_publish(
        "config/settings.yaml",
        pipeline=fake_pipeline,
        publish=lambda path: published.append(path),
    )
    assert published == ["config/settings.yaml"]

def test_refresh_and_publish_skips_upload_when_pipeline_fails(tmp_path) -> None:
    published: list[str] = []

    def fake_pipeline(config_path, **kwargs):
        raise PipelineStepError(
            name="build_metrics",
            module="pipeline.transform.build_metrics",
            returncode=2,
            remaining=["train_win_model"],
        )

    with pytest.raises(PipelineStepError):
        refresh_and_publish(
            "config/settings.yaml",
            pipeline=fake_pipeline,
            publish=lambda path: published.append(path),
        )
    assert published == []

def test_workflow_schedules_2am_mountain_and_manual_trigger() -> None:
    text = Path(".github/workflows/nightly-refresh.yml").read_text(encoding="utf-8")
    assert 'cron: "0 8 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "python3 -m pipeline.run_nightly" in text
    assert "actions/upload-artifact" in text
    assert "artifacts/extract_report.json" in text
    assert "data/raw/sportsdataio/extract_report/**" in text
    assert "ARTIFACTS_URI: ${{ secrets.ARTIFACTS_URI }}" in text
    assert "AWS_ENDPOINT_URL: ${{ secrets.AWS_ENDPOINT_URL }}" in text
    assert "AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}" in text
    assert "SPORTSDATAIO_API_KEY: ${{ secrets.SPORTSDATAIO_API_KEY }}" in text
