"""Model Insights — accuracy, feature importance, prediction misses."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    load_win_model_importance,
    load_win_model_metrics,
    load_win_model_predictions,
)
from dashboard.helpers import scoreboard_html
from dashboard.theme import CRIMSON, SURFACE, TEXT_DIM
from dashboard.ui import (
    SCATTER_MARKER as _SCATTER_MARKER,
    chart as _chart,
    empty_state as _empty,
    page_header as _page_header,
    panel_head,
    show_table as _show_table,
)


def page_model_insights() -> None:
    _page_header("Model Insights")
    model_metrics_df = load_win_model_metrics()
    importance = load_win_model_importance()
    preds = load_win_model_predictions()
    if model_metrics_df is None and importance is None and preds is None:
        _empty("models")
        return

    perf_tab, feat_tab, pred_tab = st.tabs(["Performance", "Feature Importance", "Predictions"])

    with perf_tab:
        if model_metrics_df is None:
            _empty("models")
        else:
            board = []
            for _, row in model_metrics_df.iterrows():
                name = str(row.get("model", "Model"))
                mae = row.get("mae")
                r2 = row.get("r2")
                value = f"MAE {mae:.2f}" if pd.notna(mae) else "—"
                if pd.notna(r2):
                    value = f"{value} · R² {r2:.3f}"
                board.append((name, value))
            if board:
                st.markdown(scoreboard_html(board), unsafe_allow_html=True)
            cfg = {
                "model": st.column_config.TextColumn("Model"),
                "mae": st.column_config.NumberColumn("MAE (wins)", format="%.2f"),
                "r2": st.column_config.NumberColumn("R²", format="%.4f"),
                "n_rows": st.column_config.NumberColumn("N", format="%d"),
            }
            _show_table(model_metrics_df, cfg, height=180)

    with feat_tab:
        if importance is None:
            _empty("models")
        else:
            cfg = {
                "feature": st.column_config.TextColumn("Feature", width="medium"),
                "importance": st.column_config.NumberColumn("Importance", format="%.4f"),
            }
            ranked = importance.sort_values("importance", ascending=False).reset_index(drop=True)
            _show_table(ranked, cfg, height=360)
            panel_head("Top 15 features")
            fig = px.bar(
                ranked.head(15),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, SURFACE], [1, CRIMSON]],
                labels={"importance": "Importance", "feature": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            _chart(fig, height=420)

    with pred_tab:
        if preds is None:
            _empty("models")
        else:
            err_col = "absolute_error_xgb" if "absolute_error_xgb" in preds.columns else (
                "absolute_error_lr" if "absolute_error_lr" in preds.columns else None
            )
            pred_cfg = {
                "team_name": st.column_config.TextColumn("Team", width="medium"),
                "year_id": st.column_config.NumberColumn("Year", format="%d", width="small"),
                "actual_wins": st.column_config.NumberColumn("Actual W", format="%d"),
                "predicted_wins_xgb": st.column_config.NumberColumn("XGB Pred", format="%.1f"),
                "predicted_wins_lr": st.column_config.NumberColumn("LR Pred", format="%.1f"),
                "absolute_error_xgb": st.column_config.NumberColumn("XGB Error", format="%.1f"),
                "absolute_error_lr": st.column_config.NumberColumn("LR Error", format="%.1f"),
            }
            sort_df = preds.sort_values(err_col, ascending=False).reset_index(drop=True) if err_col else preds
            st.caption(f"{len(sort_df):,} predictions — sorted by largest absolute error")
            _show_table(sort_df, pred_cfg, height=480)
            if "actual_wins" in preds.columns and "predicted_wins_xgb" in preds.columns:
                panel_head("Actual vs predicted wins")
                fig = px.scatter(
                    preds,
                    x="actual_wins",
                    y="predicted_wins_xgb",
                    hover_name="team_name",
                    hover_data=["year_id"],
                    labels={"actual_wins": "Actual wins", "predicted_wins_xgb": "XGB predicted"},
                )
                lo = preds[["actual_wins", "predicted_wins_xgb"]].min().min() - 2
                hi = preds[["actual_wins", "predicted_wins_xgb"]].max().max() + 2
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color=TEXT_DIM), name="Perfect"))
                fig.update_traces(marker=_SCATTER_MARKER, selector=dict(mode="markers"))
                _chart(fig, height=400)
