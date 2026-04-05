"""
Team clustering model.

Uses KMeans on payroll, WAR, and efficiency features to identify team archetypes:
  Big-Spend Contenders, Low-Spend Overachievers, Rebuilding Teams, Declining Spenders.

Cluster labels are determined post-hoc by centroid interpretation.
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

_CLUSTER_FEATURES = [
    "payroll",
    "wins",
    "run_diff",
    "team_total_war",
    "gini_salary",
    "top_5_salary_share",
    "wins_per_10m",
    "war_per_1m",
    "dead_money_share",
]


def _label_clusters(df: pd.DataFrame, n_clusters: int) -> dict[int, str]:
    """
    Derive cluster labels from centroid characteristics.

    Heuristic rules applied to mean values per cluster:
    - High payroll + high wins → "Big-Spend Contender"
    - Low payroll + high wins_per_10m → "Low-Spend Overachiever"
    - Low wins + low payroll → "Rebuilding"
    - High payroll + low wins → "Declining/Wasteful Spender"
    - Otherwise → "Steady Mid-Market"
    """
    summary = df.groupby("cluster_id").agg(
        payroll=("payroll", "median"),
        wins=("wins", "median"),
        wins_per_10m=("wins_per_10m", "median"),
        war_per_1m=("war_per_1m", "median"),
    )

    payroll_med = summary["payroll"].median()
    wins_med = summary["wins"].median()

    labels: dict[int, str] = {}
    used: set[str] = set()

    candidates = {
        "Big-Spend Contender": (summary["payroll"] > payroll_med) & (summary["wins"] > wins_med),
        "Low-Spend Overachiever": (summary["payroll"] <= payroll_med) & (summary["wins_per_10m"] > summary["wins_per_10m"].median()),
        "Rebuilding": (summary["wins"] < wins_med - 4) & (summary["payroll"] <= payroll_med),
        "Declining Spender": (summary["payroll"] > payroll_med) & (summary["wins"] < wins_med),
    }
    default = "Steady Mid-Market"

    # Assign each cluster to the first matching candidate; avoid duplicates
    for cid in summary.index:
        assigned = default
        for label, mask in candidates.items():
            if mask.get(cid, False) and label not in used:
                assigned = label
                used.add(label)
                break
        labels[cid] = assigned

    return labels


def _plot_clusters(df: pd.DataFrame, out_path: Path) -> None:
    palette = {
        "Big-Spend Contender": "#1f77b4",
        "Low-Spend Overachiever": "#2ca02c",
        "Rebuilding": "#ff7f0e",
        "Declining Spender": "#d62728",
        "Steady Mid-Market": "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(11, 7))
    for label, grp in df.groupby("cluster_label"):
        color = palette.get(str(label), "#8c564b")
        ax.scatter(
            grp["payroll"] / 1_000_000,
            grp["wins"],
            label=str(label),
            alpha=0.6,
            s=28,
            color=color,
        )
    ax.set_xlabel("Payroll ($M)")
    ax.set_ylabel("Wins")
    ax.set_title("Team Archetypes: Payroll vs Wins by Cluster")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    df = con.execute(
        """
        SELECT
            s.year_id, t.team_name, t.team_id, t.league_id,
            f.payroll, f.wins, f.run_diff,
            f.team_total_war, f.gini_salary, f.top_5_salary_share,
            f.wins_per_10m, f.war_per_1m, f.dead_money_share,
            f.surplus_value, f.window_phase
        FROM fact_team_season f
        JOIN dim_team t USING (team_key)
        JOIN dim_season s USING (season_key)
        """
    ).fetchdf()
    con.close()

    feature_cols = [c for c in _CLUSTER_FEATURES if c in df.columns]
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X = prep.fit_transform(df[feature_cols])

    n_clusters = settings["modeling"]["n_clusters"]
    rs = settings["modeling"]["random_state"]
    log.info("Fitting KMeans with %d clusters", n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, random_state=rs, n_init=30)
    df["cluster_id"] = kmeans.fit_predict(X)

    cluster_labels = _label_clusters(df, n_clusters)
    df["cluster_label"] = df["cluster_id"].map(cluster_labels)
    log.info("Cluster assignment:\n%s", df["cluster_label"].value_counts().to_string())

    df.to_csv(artifacts_dir / "team_clusters.csv", index=False)

    # Cluster summary stats
    summary = (
        df.groupby("cluster_label")
        .agg(
            count=("wins", "count"),
            avg_wins=("wins", "mean"),
            avg_payroll_m=("payroll", lambda x: x.mean() / 1_000_000),
            avg_war=("team_total_war", "mean"),
            avg_wins_per_10m=("wins_per_10m", "mean"),
        )
        .round(2)
        .reset_index()
    )
    summary.to_csv(artifacts_dir / "team_cluster_summary.csv", index=False)
    log.info("Cluster summary:\n%s", summary.to_string(index=False))

    _plot_clusters(df, artifacts_dir / "team_clusters_scatter.png")
    log.info("Saved cluster scatter plot")

    typer.echo("Clustering complete")


if __name__ == "__main__":
    app()
