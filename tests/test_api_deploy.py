"""Bind + deploy-config contract for the public /v1 host (fixture lake)."""
from __future__ import annotations

from pathlib import Path

import pytest

from services.api.app import bind_host, bind_port

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_LAKE = "tests/fixtures/api/lake_current"


def test_bind_port_honors_api_port_then_paas_port() -> None:
    assert bind_port({}) == 8000
    assert bind_port({"PORT": "8080"}) == 8080
    assert bind_port({"API_PORT": "9000", "PORT": "8080"}) == 9000
    assert bind_port({"API_PORT": "nope", "PORT": "8080"}) == 8000


def test_bind_host_opens_all_interfaces_when_paas_port_set() -> None:
    assert bind_host({}) == "127.0.0.1"
    assert bind_host({"PORT": "8080"}) == "0.0.0.0"
    assert bind_host({"API_HOST": "127.0.0.1", "PORT": "8080"}) == "127.0.0.1"
    assert bind_host({"API_HOST": "0.0.0.0"}) == "0.0.0.0"


def test_deploy_files_point_at_fixture_lake_and_vercel_cors() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    railway = (ROOT / "railway.json").read_text(encoding="utf-8")
    procfile = (ROOT / "Procfile").read_text(encoding="utf-8")
    slim = (ROOT / "requirements-api.txt").read_text(encoding="utf-8")

    assert FIXTURE_LAKE in dockerfile
    assert "file:///app/tests/fixtures/api/lake_current" in dockerfile
    assert "API_CORS_ORIGINS=*" in dockerfile
    assert "API_CORS_ORIGIN_REGEX=https://.*[.]vercel[.]app" in dockerfile
    assert "API_HOST=0.0.0.0" in dockerfile
    assert "ENV API_PORT" not in dockerfile
    assert "CMD [\"python3\", \"-m\", \"services.api\"]" in dockerfile
    assert "/v1/teams" not in dockerfile

    assert FIXTURE_LAKE in render
    assert 'value: "*"' in render
    assert "https://.*[.]vercel[.]app" in render
    assert "/v1/teams" not in render

    assert "DOCKERFILE" in railway
    assert "/v1/health" in railway
    assert "python3 -m services.api" in procfile
    assert "fastapi" in slim
    assert "streamlit" not in slim
    assert "xgboost" not in slim
