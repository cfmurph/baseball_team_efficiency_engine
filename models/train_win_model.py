"""
Win prediction model.

Trains both a baseline LinearRegression and an XGBoost model, then compares
them and writes the better model's predictions as the primary artifact.
Also generates the efficiency frontier analysis (regression envelope on
payroll vs wins).
"""
from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

try:
    from xgboost import XGBRegressor
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False)

_FEATURES = [
    "run_diff",
    "pythag_wins",
    "pythag_gap",
    "team_total_war",
    "war_win_gap",
    "payroll",
    "max_salary",
    "median_salary",
    "top_1_salary_share",
    "top_3_salary_share",
    "top_5_salary_share",
    "gini_salary",
    "dead_money_share",
    "wins_per_10m",
    "run_diff_per_10m",
    "cost_per_war",
    "war_per_1m",
    "surplus_value",
]

_PAYROLL_FEATURES = [
    "payroll",
    "team_total_war",
    "gini_salary",
    "top_5_salary_share",
    "dead_money_share",
    "war_per_1m",
]


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                feature_cols,
            )
        ]
    )


def _make_linear_pipeline(feature_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", _build_preprocessor(feature_cols)),
        ("regressor", LinearRegression()),
    ])


def _make_xgb_pipeline(feature_cols: list[str], random_state: int) -> Pipeline:
    if not _XGB_AVAILABLE:
        raise RuntimeError("xgboost is not installed. Run: pip install xgboost")
    return Pipeline([
        ("preprocessor", _build_preprocessor(feature_cols)),
        ("regressor", XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )),
    ])


def _plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    items = list(predictions.items())
    fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 6))
    if len(items) == 1:
        axes = [axes]
    for ax, (label, preds) in zip(axes, items):
        ax.scatter(y_test, preds, alpha=0.55, s=22, color="#2a7ae2")
        lo = min(y_test.min(), preds.min()) - 2
        hi = max(y_test.max(), preds.max()) + 2
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("Actual Wins")
        ax.set_ylabel("Predicted Wins")
        ax.set_title(f"{label}: Actual vs Predicted")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_efficiency_frontier(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of payroll vs wins with a polynomial efficiency frontier.
    Teams above the curve are efficient; below are inefficient.
    """
    df = df.dropna(subset=["payroll", "wins"]).copy()
    df["payroll_m"] = df["payroll"] / 1_000_000

    # Fit polynomial frontier using only top-N efficient quantile
    from numpy.polynomial import polynomial as P
    # Use 75th-percentile rolling wins as frontier proxy
    df_sorted = df.sort_values("payroll_m")
    x = df_sorted["payroll_m"].values
    y = df_sorted["wins"].values

    # Frontier: 75th percentile wins binned by payroll decile
    df_sorted["payroll_bin"] = pd.qcut(df_sorted["payroll_m"], q=10, duplicates="drop")
    frontier_pts = df_sorted.groupby("payroll_bin")["wins"].quantile(0.75).reset_index()
    fx = df_sorted.groupby("payroll_bin")["payroll_m"].mean().values
    fy = frontier_pts["wins"].values

    coeffs = np.polyfit(fx, fy, 2)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)

    # Above/below curve
    df["frontier_pred"] = np.polyval(coeffs, df["payroll_m"])
    df["above_frontier"] = df["wins"] >= df["frontier_pred"]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = df["above_frontier"].map({True: "#2ca02c", False: "#d62728"})
    ax.scatter(df["payroll_m"], df["wins"], c=colors, alpha=0.5, s=25, zorder=2)
    ax.plot(x_line, y_line, "b--", lw=2, label="Efficiency frontier (75th pct)", zorder=3)
    ax.set_xlabel("Payroll ($M)")
    ax.set_ylabel("Wins")
    ax.set_title("Efficiency Frontier: Payroll vs Wins (1990–present)")
    above_patch = mlines.Line2D([], [], color="#2ca02c", marker="o", ls="", label="Above frontier (efficient)")
    below_patch = mlines.Line2D([], [], color="#d62728", marker="o", ls="", label="Below frontier (wasteful)")
    ax.legend(handles=[above_patch, below_patch, ax.lines[0]], loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return df[["year_id", "team_name", "payroll_m", "wins", "frontier_pred", "above_frontier"]]


@app.command()
def main(config_path: str = "config/settings.yaml") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)
    con = duckdb.connect(settings["warehouse_path"])
    artifacts_dir = ensure_dir(settings["artifacts_dir"])

    df = con.execute(
        """
        SELECT
            s.year_id, t.team_name,
            f.wins, f.run_diff, f.pythag_wins, f.pythag_gap,
            f.team_total_war, f.war_win_gap,
            f.payroll, f.max_salary, f.median_salary,
            f.top_1_salary_share, f.top_3_salary_share, f.top_5_salary_share,
            f.gini_salary, f.dead_money_share,
            f.wins_per_10m, f.run_diff_per_10m,
            f.cost_per_war, f.war_per_1m, f.surplus_value
        FROM fact_team_season f
        JOIN dim_team t USING (team_key)
        JOIN dim_season s USING (season_key)
        """
    ).fetchdf()
    con.close()

    if not _XGB_AVAILABLE:
        log.warning("xgboost not installed — running LinearRegression only. Install with: pip install xgboost")

    feature_cols = [c for c in _FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df["wins"]

    rs = settings["modeling"]["random_state"]
    test_size = settings["modeling"]["test_size"]

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, df[["year_id", "team_name"]], test_size=test_size, random_state=rs
    )

    # ---- LinearRegression (always) ----
    log.info("Training LinearRegression baseline (%d train, %d test)", len(X_train), len(X_test))
    lr_model = _make_linear_pipeline(feature_cols)
    lr_model.fit(X_train, y_train)
    preds_lr = lr_model.predict(X_test)

    all_preds: dict[str, np.ndarray] = {"Linear Regression": preds_lr}
    metrics_rows = [{
        "model": "LinearRegression",
        "mae": mean_absolute_error(y_test, preds_lr),
        "r2": r2_score(y_test, preds_lr),
    }]

    # ---- XGBoost (optional) ----
    preds_xgb: np.ndarray | None = None
    xgb_model = None
    if _XGB_AVAILABLE:
        log.info("Training XGBoost model")
        xgb_model = _make_xgb_pipeline(feature_cols, rs)
        xgb_model.fit(X_train, y_train)
        preds_xgb = xgb_model.predict(X_test)
        all_preds["XGBoost"] = preds_xgb
        metrics_rows.append({
            "model": "XGBoost",
            "mae": mean_absolute_error(y_test, preds_xgb),
            "r2": r2_score(y_test, preds_xgb),
        })

    # ---- Metrics ----
    metrics = pd.DataFrame(metrics_rows)
    metrics["n_rows"] = len(df)
    metrics.to_csv(artifacts_dir / "win_model_metrics.csv", index=False)
    log.info("Model metrics:\n%s", metrics.to_string(index=False))

    # ---- Predictions CSV ----
    results = meta_test.copy()
    results["actual_wins"] = y_test.values
    results["predicted_wins_lr"] = preds_lr
    primary_err_col = "absolute_error_lr"
    if preds_xgb is not None:
        results["predicted_wins_xgb"] = preds_xgb
        results["absolute_error_xgb"] = (results["actual_wins"] - preds_xgb).abs()
        primary_err_col = "absolute_error_xgb"
    else:
        results["absolute_error_lr"] = (results["actual_wins"] - preds_lr).abs()
    results = results.sort_values(primary_err_col, ascending=False)
    results.to_csv(artifacts_dir / "win_model_predictions.csv", index=False)

    # ---- Feature importance (XGBoost preferred, LR fallback via coef) ----
    if xgb_model is not None:
        xgb_reg = xgb_model.named_steps["regressor"]
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": xgb_reg.feature_importances_,
        }).sort_values("importance", ascending=False)
    else:
        lr_reg = lr_model.named_steps["regressor"]
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": np.abs(lr_reg.coef_),
        }).sort_values("importance", ascending=False)
    importances.to_csv(artifacts_dir / "win_model_feature_importance.csv", index=False)
    log.info("Top features:\n%s", importances.head(10).to_string(index=False))

    # ---- Plots ----
    _plot_actual_vs_predicted(
        y_test.values,
        all_preds,
        artifacts_dir / "win_model_actual_vs_predicted.png",
    )
    log.info("Saved actual vs predicted plot")

    frontier_df = _plot_efficiency_frontier(df, artifacts_dir / "win_model_efficiency_frontier.png")
    if frontier_df is not None:
        frontier_df.to_csv(artifacts_dir / "win_model_frontier_data.csv", index=False)
    log.info("Saved efficiency frontier plot")

    typer.echo("Model training complete")


if __name__ == "__main__":
    app()
