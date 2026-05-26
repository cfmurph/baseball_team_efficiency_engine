from __future__ import annotations

import datetime
import importlib.util
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ColumnConfig:
    @staticmethod
    def NumberColumn(label: str, **kwargs):
        return {"type": "number", "label": label, **kwargs}

    @staticmethod
    def TextColumn(label: str, **kwargs):
        return {"type": "text", "label": label, **kwargs}

    @staticmethod
    def CheckboxColumn(label: str, **kwargs):
        return {"type": "checkbox", "label": label, **kwargs}


class _Sidebar:
    def markdown(self, *args, **kwargs) -> None:
        return None

    def radio(self, label, options, **kwargs):
        return options[-1]


class _FakeStreamlit:
    def __init__(self) -> None:
        self.column_config = _ColumnConfig()
        self.sidebar = _Sidebar()
        self.session_state = {}
        self.plotly_chart_calls = []

    def cache_data(self, *args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda func: func

    def set_page_config(self, *args, **kwargs) -> None:
        return None

    def markdown(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def stop(self) -> None:
        raise RuntimeError("streamlit stop called during dashboard import")

    def title(self, *args, **kwargs) -> None:
        return None

    def caption(self, *args, **kwargs) -> None:
        return None

    def tabs(self, labels):
        return [_Context() for _ in labels]

    def dataframe(self, *args, **kwargs) -> None:
        return None

    def plotly_chart(self, *args, **kwargs) -> None:
        self.plotly_chart_calls.append((args, kwargs))


def test_scale_payroll_converts_display_columns_without_mutating_input(monkeypatch, tmp_path) -> None:
    module, _ = _import_dashboard(monkeypatch, tmp_path)
    raw = pd.DataFrame(
        {
            "payroll": [80_000_000],
            "salary": [5_500_000],
            "surplus_value": [12_000_000],
            "dead_money_share": [0.125],
            "wins": [92],
        }
    )

    scaled = module._scale_payroll(raw)

    assert scaled.loc[0, "payroll"] == 80.0
    assert scaled.loc[0, "salary"] == 5.5
    assert scaled.loc[0, "surplus_value"] == 12.0
    assert scaled.loc[0, "dead_money_share"] == 12.5
    assert scaled.loc[0, "wins"] == 92
    assert raw.loc[0, "payroll"] == 80_000_000


def test_chart_applies_layout_and_calls_streamlit_plotly_chart(monkeypatch, tmp_path) -> None:
    module, fake_st = _import_dashboard(monkeypatch, tmp_path)
    fig = go.Figure()

    module._chart(fig, height=321)

    assert len(fake_st.plotly_chart_calls) == 1
    args, kwargs = fake_st.plotly_chart_calls[0]
    assert args == (fig,)
    assert kwargs == {"use_container_width": True}
    assert fig.layout.height == 321
    assert fig.layout.paper_bgcolor == "#0d1117"


def test_empty_metrics_years_use_current_year_for_slider_max(monkeypatch, tmp_path) -> None:
    empty_metrics = pd.DataFrame(columns=["year_id", "team_name"])

    module, _ = _import_dashboard(monkeypatch, tmp_path, metrics=empty_metrics)

    assert module.all_years == []
    assert module._slider_max == datetime.date.today().year


def _import_dashboard(monkeypatch, tmp_path, metrics: pd.DataFrame | None = None):
    metrics = metrics if metrics is not None else _default_metrics()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    metrics.to_csv(artifacts / "team_onfield_contract_metrics.csv", index=False)
    monkeypatch.chdir(tmp_path)

    fake_st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    module_name = f"dashboard_app_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "dashboard" / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module, fake_st


def _default_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year_id": [2024],
            "team_name": ["Aces"],
            "wins": [92],
            "payroll": [80_000_000],
            "run_diff": [120],
            "wins_per_10m": [11.5],
        }
    )
