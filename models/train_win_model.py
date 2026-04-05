from __future__ import annotations

from pathlib import Path
import duckdb
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

app = typer.Typer(add_completion=False)


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    df = con.execute(
        """
        SELECT year_id, team_name, wins, run_diff, pythag_wins, payroll,
               max_salary, median_salary, top_1_salary_share, top_3_salary_share,
               top_5_salary_share, gini_salary, payroll_per_win, wins_per_10m, run_diff_per_10m
        FROM fact_team_season
        JOIN dim_team USING(team_key)
        JOIN dim_season USING(season_key)
        """
    ).fetchdf()
    con.close()

    feature_cols = [
        "run_diff", "pythag_wins", "payroll", "max_salary", "median_salary",
        "top_1_salary_share", "top_3_salary_share", "top_5_salary_share", "gini_salary",
        "payroll_per_win", "wins_per_10m", "run_diff_per_10m"
    ]

    X = df[feature_cols]
    y = df["wins"]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df[["year_id", "team_name"]], test_size=settings["modeling"]["test_size"], random_state=settings["modeling"]["random_state"]
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, feature_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=300, random_state=settings["modeling"]["random_state"])),
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results = meta_test.copy()
    results["actual_wins"] = y_test.values
    results["predicted_wins"] = preds
    results["absolute_error"] = (results["actual_wins"] - results["predicted_wins"]).abs()
    results = results.sort_values("absolute_error", ascending=False)
    results.to_csv(Path(artifacts_dir) / "win_model_predictions.csv", index=False)

    metrics = pd.DataFrame([{
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "n_rows": len(df),
    }])
    metrics.to_csv(Path(artifacts_dir) / "win_model_metrics.csv", index=False)

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.named_steps["regressor"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(Path(artifacts_dir) / "win_model_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title("Win Model: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(Path(artifacts_dir) / "win_model_actual_vs_predicted.png", dpi=150)
    plt.close()

    typer.echo("Model training complete")


if __name__ == "__main__":
    app()
