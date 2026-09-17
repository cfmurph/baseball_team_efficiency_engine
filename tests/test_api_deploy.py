"""Bind + deploy-config contract for the public /v1 host (fixture lake)."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from services.api.app import (
    DEFAULT_CORS_ORIGINS,
    _install_secret_log_filter,
    bind_host,
    bind_port,
    cors_origin_regex,
    cors_origins,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_LAKE = "tests/fixtures/api/lake_current"
SECRET = "super-secret-sdio-key-not-for-ci"


def test_bind_port_honors_api_port_then_paas_port() -> None:
    assert bind_port({}) == 8000
    assert bind_port({"PORT": "8080"}) == 8080
    assert bind_port({"API_PORT": "9000", "PORT": "8080"}) == 9000
    assert bind_port({"API_PORT": "nope", "PORT": "8080"}) == 8000
    # Whitespace-only API_PORT is truthy before strip, so PORT is not used.
    assert bind_port({"API_PORT": "  ", "PORT": "8080"}) == 8000
    assert bind_port({"API_PORT": "", "PORT": "8080"}) == 8080
    assert bind_port({"API_PORT": "", "PORT": ""}) == 8000


def test_bind_host_opens_all_interfaces_when_paas_port_set() -> None:
    assert bind_host({}) == "127.0.0.1"
    assert bind_host({"PORT": "8080"}) == "0.0.0.0"
    assert bind_host({"API_HOST": "127.0.0.1", "PORT": "8080"}) == "127.0.0.1"
    assert bind_host({"API_HOST": "0.0.0.0"}) == "0.0.0.0"
    assert bind_host({"API_HOST": "  ", "PORT": "8080"}) == "0.0.0.0"


def test_cors_origins_parses_wildcard_csv_and_defaults() -> None:
    assert cors_origins({}) == list(DEFAULT_CORS_ORIGINS)
    assert cors_origins({"API_CORS_ORIGINS": ""}) == list(DEFAULT_CORS_ORIGINS)
    assert cors_origins({"API_CORS_ORIGINS": "*"}) == ["*"]
    assert cors_origins({"API_CORS_ORIGINS": "all"}) == ["*"]
    assert cors_origins(
        {"API_CORS_ORIGINS": " https://a.example , , https://b.example "}
    ) == ["https://a.example", "https://b.example"]


def test_cors_origin_regex_treats_blank_as_unset() -> None:
    # Empty regex would compile to match-all in Starlette; blank must be None.
    assert cors_origin_regex({}) is None
    assert cors_origin_regex({"API_CORS_ORIGIN_REGEX": ""}) is None
    assert cors_origin_regex({"API_CORS_ORIGIN_REGEX": "   "}) is None
    assert (
        cors_origin_regex({"API_CORS_ORIGIN_REGEX": "https://.*[.]vercel[.]app"})
        == "https://.*[.]vercel[.]app"
    )


def test_secret_log_filter_redacts_vendor_keys_and_clears_args() -> None:
    env = {"SPORTSDATAIO_API_KEY": SECRET}
    root = logging.getLogger()
    before_ids = {id(item) for item in root.filters}
    try:
        _install_secret_log_filter(env)
        added = [item for item in root.filters if id(item) not in before_ids]
        assert added
        record = logging.LogRecord(
            name="services.api",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="using key=%s",
            args=(SECRET,),
            exc_info=None,
        )
        assert added[-1].filter(record) is True
        text = record.getMessage()
        assert SECRET not in text
        assert "[SPORTSDATAIO_API_KEY]" in text
        assert record.args == ()
    finally:
        for item in list(root.filters):
            if id(item) not in before_ids:
                root.removeFilter(item)


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
