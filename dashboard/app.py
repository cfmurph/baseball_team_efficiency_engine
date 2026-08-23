"""
MLB Team Efficiency Engine — Front-office dashboard

Entrypoint: ``streamlit run dashboard/app.py`` (from repo root).

Pages call named loaders in ``dashboard.data`` — never raw artifact paths.
Shared session keys (``dashboard.state``): season_year, selected_team,
selected_league. Artifact resolution uses ``resolve_artifact()``
(ARTIFACTS_URI → local artifacts/ fallback).
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="EE · Front Office",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.baseball_analytics.config import load_artifact_settings
from src.baseball_analytics.dashboard_helpers import compute_slider_max
from src.baseball_analytics.dashboard_utils import (
    player_id_columns_for_duplicate_names,
    scale_payroll_for_display,
)
from src.baseball_analytics.storage import (
    artifact_source_label,
    resolve_artifact,
)
from dashboard.data import (
    ARTIFACT_NAMES,
    load_team_metrics,
    resolve_file,
    resolve_all,
)
from dashboard.helpers import (
    artifact_status,
    nav_labels,
    teams_from_frame,
    years_from_frame,
)
from dashboard.views import compare as compare_view
from dashboard.views import contracts as contracts_view
from dashboard.views import deep_dive as deep_dive_view
from dashboard.views import frontier as frontier_view
from dashboard.views import models as models_view
from dashboard.views import overview as overview_view
from dashboard.views import roster as roster_view
from dashboard.views import whatif as whatif_view

# Bind the chrome module by path so a merge cannot leave `ui` undefined.
import dashboard.ui as ui
from dashboard.ui import SCATTER_MARKER

ui.inject_theme()

# Literal layout so AST regression tests can exec `_chart` without imports.
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Sans, sans-serif", color="#9aa4b2", size=11),
    margin=dict(t=20, b=36, l=44, r=12),
    colorway=["#ff2d3a", "#6ecbff", "#3ee08f", "#f5c518", "#b794f6", "#ff7a59"],
)
_SCATTER_MARKER = SCATTER_MARKER


def _apply_layout(fig) -> None:
    fig.update_layout(**_PLOTLY_LAYOUT)


def _chart(fig, height: int = 400) -> None:
    _apply_layout(fig)
    fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True)


_ARTIFACT_SETTINGS = load_artifact_settings(str(_ROOT / "config/settings.yaml"))
ARTIFACTS = _ARTIFACT_SETTINGS.local_dir
_FILES = {key: ARTIFACTS / name for key, name in ARTIFACT_NAMES.items()}


def _resolve_file(key: str) -> str | None:
    path = resolve_file(key, _ARTIFACT_SETTINGS)
    return None if path is None else str(path)


def _load(key: str) -> pd.DataFrame | None:
    raw = _resolve_file(key)
    if raw is None:
        return None
    return pd.read_csv(raw)


metrics = load_team_metrics()
_current_year = datetime.date.today().year
all_years = years_from_frame(metrics)
_slider_max = compute_slider_max(all_years, _current_year)
_slider_lo = all_years[0] if all_years else _current_year
all_teams = teams_from_frame(metrics)
_status = artifact_status(resolve_all(_ARTIFACT_SETTINGS))
_source_label = artifact_source_label(_ARTIFACT_SETTINGS)

ui.ALL_YEARS = all_years
overview_view.metrics = metrics
overview_view.all_years = all_years
deep_dive_view.metrics = metrics
deep_dive_view.all_teams = all_teams
deep_dive_view.all_years = all_years
compare_view.metrics = metrics
compare_view.all_teams = all_teams
compare_view._slider_lo = _slider_lo
compare_view._slider_max = _slider_max
whatif_view.metrics = metrics
whatif_view.all_teams = all_teams
whatif_view.all_years = all_years


def _page_header(label: str) -> None:
    ui.page_header(label)


def _empty(kind: str) -> None:
    ui.empty_state(kind)


def _salary_note(year: int | None) -> None:
    ui.salary_note(year)


def _season_picker(key: str = "season", default_latest: bool = True) -> int | None:
    return ui.season_picker(key, default_latest=default_latest)


def page_league_snapshot() -> None:
    overview_view.page_league_snapshot()


def page_player_explorer() -> None:
    roster_view.page_player_explorer()


def page_team_deep_dive() -> None:
    deep_dive_view.page_team_deep_dive()


def page_compare_teams() -> None:
    compare_view.page_compare_teams()


def page_contract_analysis() -> None:
    contracts_view.page_contract_analysis()


def page_efficiency_frontier() -> None:
    frontier_view.page_efficiency_frontier()


def page_whatif() -> None:
    whatif_view.page_whatif()


def page_model_insights() -> None:
    models_view.page_model_insights()


page_overview = page_league_snapshot
page_roster_lab = page_player_explorer
page_contract_watch = page_contract_analysis
page_team_profile = page_team_deep_dive
page_season_compare = page_compare_teams

_PAGES = {
    "Overview": page_overview,
    "Team Deep Dive": page_team_deep_dive,
    "Compare Teams": page_compare_teams,
    "Roster Lab": page_roster_lab,
    "Contract Watch": page_contract_watch,
    "Efficiency Frontier": page_efficiency_frontier,
    "What-If Sim": page_whatif,
    "Model Insights": page_model_insights,
}

page = ui.render_sidebar(all_years=all_years, status=_status, source=_source_label)
ui.render_app_frame(all_years=all_years, status=_status, source=_source_label, page=page)
# nav_labels() is the product section list; keep it imported for tests/helpers.
assert page in nav_labels() or page in _PAGES
_PAGES[page]()
