"""Guard the PR/push CI smoke workflow stays offline and complete."""
from __future__ import annotations

from pathlib import Path

WORKFLOW = Path(".github/workflows/ci-smoke.yml")


def test_ci_smoke_runs_on_pull_request_and_push_to_master() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" in text
    assert "push:" in text
    assert "branches: [master]" in text
    assert 'python3 -m pytest tests/ -m "not network" -v' in text
    assert "nightly-refresh.yml" not in text


def test_ci_smoke_does_not_invoke_live_extracts() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pipeline.extract.pull_sources" not in text
    assert "pipeline.extract.pull_war" not in text
    assert "pipeline.run_nightly" not in text
