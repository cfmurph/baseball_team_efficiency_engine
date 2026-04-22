from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


class _NoOpContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSidebar:
    def __init__(self, radio_choice: str):
        self._radio_choice = radio_choice

    def markdown(self, *args, **kwargs) -> None:
        return None

    def radio(self, _label, options, **kwargs):
        if self._radio_choice in options:
            return self._radio_choice
        return options[0]


class _FakeStreamlit:
    def __init__(self, radio_choice: str):
        self.session_state = {}
        self.sidebar = _FakeSidebar(radio_choice)
        self.column_config = SimpleNamespace(
            NumberColumn=lambda *args, **kwargs: ("number", args, kwargs),
            TextColumn=lambda *args, **kwargs: ("text", args, kwargs),
            CheckboxColumn=lambda *args, **kwargs: ("checkbox", args, kwargs),
        )

    def cache_data(self, ttl=None):  # noqa: ARG002
        def decorator(func):
            return func

        return decorator

    def set_page_config(self, *args, **kwargs) -> None:
        return None

    def markdown(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def stop(self) -> None:
        raise RuntimeError("st.stop should not be reached in tests")

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [_NoOpContext() for _ in range(count)]

    def button(self, *args, **kwargs):
        return False

    def selectbox(self, _label, options, index=0, **kwargs):  # noqa: ARG002
        options = list(options)
        if not options:
            return None
        if index < 0:
            index = 0
        if index >= len(options):
            index = len(options) - 1
        return options[index]

    def multiselect(self, _label, options, default=None, **kwargs):  # noqa: ARG002
        if default is not None:
            return list(default)
        return []

    def text_input(self, *args, **kwargs):
        return kwargs.get("value", "")

    def slider(self, _label, _min_value, _max_value, value, **kwargs):  # noqa: ARG002
        return value

    def tabs(self, labels):
        return [_NoOpContext() for _ in labels]

    def expander(self, *args, **kwargs):
        return _NoOpContext()

    def dataframe(self, *args, **kwargs) -> None:
        return None

    def plotly_chart(self, *args, **kwargs) -> None:
        return None

    def title(self, *args, **kwargs) -> None:
        return None

    def caption(self, *args, **kwargs) -> None:
        return None

    def subheader(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def info(self, *args, **kwargs) -> None:
        return None

    def divider(self, *args, **kwargs) -> None:
        return None

    def metric(self, *args, **kwargs) -> None:
        return None


def _load_dashboard_app(tmp_path: Path, monkeypatch, metrics_df: pd.DataFrame):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(artifacts / "team_onfield_contract_metrics.csv", index=False)

    fake_st = _FakeStreamlit(radio_choice="👤  Player Explorer")
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    monkeypatch.chdir(tmp_path)

    module_name = f"dashboard_app_test_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_chart_applies_layout_sets_height_and_renders(monkeypatch, tmp_path):
    metrics_df = pd.DataFrame(
        [{"year_id": 2024, "team_name": "Yankees", "wins": 90, "losses": 72, "payroll": 200_000_000}]
    )
    app = _load_dashboard_app(tmp_path, monkeypatch, metrics_df)

    fig = go.Figure()
    calls = {"layout": 0, "plotly_chart": 0}

    def _fake_apply_layout(got_fig):
        calls["layout"] += 1
        assert got_fig is fig

    def _fake_plotly_chart(got_fig, use_container_width):
        calls["plotly_chart"] += 1
        assert got_fig is fig
        assert use_container_width is True

    monkeypatch.setattr(app, "_apply_layout", _fake_apply_layout)
    monkeypatch.setattr(app.st, "plotly_chart", _fake_plotly_chart)

    app._chart(fig, height=321)

    assert calls["layout"] == 1
    assert calls["plotly_chart"] == 1
    assert fig.layout.height == 321


def test_slider_max_defaults_to_current_year_when_all_years_empty(monkeypatch, tmp_path):
    metrics_df = pd.DataFrame(columns=["year_id", "team_name", "wins", "losses", "payroll"])
    app = _load_dashboard_app(tmp_path, monkeypatch, metrics_df)

    assert app.all_years == []
    assert app._slider_max == app._current_year


def test_slider_max_uses_latest_metric_year_when_newer(monkeypatch, tmp_path):
    current_year = pd.Timestamp.today().year
    next_year = current_year + 1
    metrics_df = pd.DataFrame(
        [{"year_id": next_year, "team_name": "Yankees", "wins": 90, "losses": 72, "payroll": 200_000_000}]
    )
    app = _load_dashboard_app(tmp_path, monkeypatch, metrics_df)

    assert app.all_years == [next_year]
    assert app._slider_max == next_year
