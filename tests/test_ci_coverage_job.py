"""Lock the report-only Coverage CI job that #149 claimed but never landed."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

CI_YML = Path(".github/workflows/ci.yml")
TESTING_DOC = Path("docs/testing.md")


def test_coverage_job_is_report_only_and_off_the_smoke_alias() -> None:
    text = CI_YML.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    jobs = workflow["jobs"]

    assert "coverage" in jobs
    coverage = jobs["coverage"]
    assert coverage["name"] == "Coverage"
    assert "needs" not in coverage

    run_text = "\n".join(
        step.get("run", "") for step in coverage["steps"] if isinstance(step, dict)
    )
    assert "--cov=src" in run_text
    assert "--cov=pipeline" in run_text
    assert "--cov=dashboard" in run_text
    assert "--cov=services" in run_text
    assert "--cov=fantasy" in run_text
    assert "--cov-fail-under" not in run_text
    assert "GITHUB_STEP_SUMMARY" in run_text
    assert "coverage report" in run_text
    assert "ARTIFACTS_URI" not in run_text
    assert "codecov" not in text.lower()

    uploads = [
        step
        for step in coverage["steps"]
        if isinstance(step, dict)
        and str(step.get("uses", "")).startswith("actions/upload-artifact")
    ]
    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert "coverage.xml" in upload["path"]
    assert "htmlcov" in upload["path"]
    assert int(upload["retention-days"]) <= 7

    smoke = jobs["smoke_alias"]
    assert smoke["needs"] == ["unit", "integration", "e2e"]
    assert "coverage" not in smoke["needs"]

    docs = TESTING_DOC.read_text(encoding="utf-8")
    assert "--cov=src --cov=pipeline --cov=dashboard --cov=services --cov=fantasy" in docs
    assert "--cov-fail-under" in docs
    assert "Do not add Coverage to `smoke_alias.needs`" in docs
