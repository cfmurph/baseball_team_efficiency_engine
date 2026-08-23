"""Tests for the nightly refresh orchestrator."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.run_nightly import (
    PIPELINE_STEPS,
    PipelineStepError,
    _step_command,
    run_pipeline,
)


def test_pipeline_steps_match_documented_chain() -> None:
    assert [module for _, module in PIPELINE_STEPS] == [
        "pipeline.extract.pull_sources",
        "pipeline.extract.pull_war",
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
        "pipeline.transform.build_warehouse",
        "pipeline.transform.build_metrics",
    ]
    assert "Not run: train_win_model, cluster_teams" in str(err)


def test_workflow_schedules_2am_mountain_and_manual_trigger() -> None:
    text = Path(".github/workflows/nightly-refresh.yml").read_text(encoding="utf-8")
    assert 'cron: "0 8 * * *"' in text
    assert "workflow_dispatch:" in text
    assert "python3 -m pipeline.run_nightly" in text
    assert "actions/upload-artifact" in text
