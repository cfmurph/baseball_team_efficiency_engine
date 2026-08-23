"""Thin data-access layer for dashboard pages.

Pages call named loaders only — never ``Path("artifacts")`` or raw CSV names.

Resolution is ``resolve_artifact()`` from ``src.baseball_analytics.storage``
(#105 / ADR 0001 contract):

1. Fresh disk cache under ``artifacts/.remote_cache/`` (TTL)
2. ``current/<metrics|models|fantasy>/<file>`` when ``ARTIFACTS_URI`` is set
3. Deprecated one-release compat: ``{league}/{level}/latest/<file>``
4. Stale remote cache if the store is unreachable
5. Local ``artifacts/<file>``
6. Missing → ``None`` (pages show the existing empty-state card)

``ARTIFACTS_URI`` env overrides ``artifacts_uri`` in ``config/settings.yaml``.
Schemes: ``s3://``, ``r2://``, ``gs://``, ``file://``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.baseball_analytics.config import ArtifactSettings, load_artifact_settings
from src.baseball_analytics.storage import artifact_source_label, resolve_artifact

# Logical name → filename. Paths are resolved; pages never see these strings.
ARTIFACT_NAMES: dict[str, str] = {
    "metrics": "team_onfield_contract_metrics.csv",
    "frontier": "team_efficiency_frontier.csv",
    "clusters": "team_clusters.csv",
    "cluster_summ": "team_cluster_summary.csv",
    "players": "player_season_metrics.csv",
    "top_value": "player_top_surplus_value.csv",
    "worst": "player_worst_contracts.csv",
    "dead": "player_dead_money.csv",
    "preds": "win_model_predictions.csv",
    "importance": "win_model_feature_importance.csv",
    "model_metrics": "win_model_metrics.csv",
    "window": "team_window_phases.csv",
    "frontier_data": "win_model_frontier_data.csv",
    "sr_players": "sr_player_season_metrics.csv",
    "sr_injuries": "sr_injuries.csv",
    "sr_tx": "sr_transactions.csv",
}


def artifact_settings() -> ArtifactSettings:
    return load_artifact_settings("config/settings.yaml")


def source_label(settings: ArtifactSettings | None = None) -> str:
    """Sidebar Source: ``remote`` | ``local`` | ``missing``."""
    return artifact_source_label(settings or artifact_settings())


def resolve_file(key: str, settings: ArtifactSettings | None = None) -> Path | None:
    """Resolve one logical artifact to a local readable path, or None."""
    name = ARTIFACT_NAMES.get(key)
    if name is None:
        return None
    return resolve_artifact(name, settings or artifact_settings())


def resolve_all(settings: ArtifactSettings | None = None) -> dict[str, Path | None]:
    cfg = settings or artifact_settings()
    return {key: resolve_file(key, cfg) for key in ARTIFACT_NAMES}


@st.cache_data(ttl=300)
def _read_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def load_named_artifact(key: str) -> pd.DataFrame | None:
    """Load a logical artifact by key (``metrics``, ``players``, …)."""
    path = resolve_file(key)
    if path is None:
        return None
    return _read_csv(str(path))


def load_team_metrics() -> pd.DataFrame | None:
    return load_named_artifact("metrics")


def load_player_season_metrics() -> pd.DataFrame | None:
    return load_named_artifact("players")


def load_window_phases() -> pd.DataFrame | None:
    return load_named_artifact("window")


def load_frontier_data() -> pd.DataFrame | None:
    return load_named_artifact("frontier_data")


def load_team_clusters() -> pd.DataFrame | None:
    return load_named_artifact("clusters")


def load_cluster_summary() -> pd.DataFrame | None:
    return load_named_artifact("cluster_summ")


def load_win_model_metrics() -> pd.DataFrame | None:
    return load_named_artifact("model_metrics")


def load_win_model_importance() -> pd.DataFrame | None:
    return load_named_artifact("importance")


def load_win_model_predictions() -> pd.DataFrame | None:
    return load_named_artifact("preds")


def load_sr_player_metrics() -> pd.DataFrame | None:
    return load_named_artifact("sr_players")
