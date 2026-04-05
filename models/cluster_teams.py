from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import typer
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

app = typer.Typer(add_completion=False)

CLUSTER_LABELS = {
    0: "Balanced",
    1: "High Spend",
    2: "Efficient Low Spend",
    3: "Underperforming Spend",
}


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    df = con.execute(
        """
        SELECT year_id, team_name, payroll, wins, run_diff, gini_salary, top_5_salary_share, wins_per_10m
        FROM fact_team_season
        JOIN dim_team USING(team_key)
        JOIN dim_season USING(season_key)
        """
    ).fetchdf()
    con.close()

    features = ["payroll", "wins", "run_diff", "gini_salary", "top_5_salary_share", "wins_per_10m"]
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X = pipeline.fit_transform(df[features])

    kmeans = KMeans(n_clusters=settings["modeling"]["n_clusters"], random_state=settings["modeling"]["random_state"], n_init=20)
    df["cluster_id"] = kmeans.fit_predict(X)
    df["cluster_label"] = df["cluster_id"].map(CLUSTER_LABELS).fillna("Unlabeled")
    df.to_csv(Path(artifacts_dir) / "team_clusters.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(df["payroll"], df["wins"])
    plt.xlabel("Payroll")
    plt.ylabel("Wins")
    plt.title("Payroll vs Wins by Cluster")
    plt.tight_layout()
    plt.savefig(Path(artifacts_dir) / "team_clusters_scatter.png", dpi=150)
    plt.close()

    typer.echo("Clustering complete")


if __name__ == "__main__":
    app()
