"""Pure helpers for the Streamlit dashboard. No Streamlit imports."""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd

SALARY_DATA_LAST_YEAR = 2016

MONEY_COLS_TO_MILLIONS = (
    "payroll",
    "max_salary",
    "median_salary",
    "payroll_per_win",
    "cost_per_war",
    "surplus_value",
    "salary",
)

METRIC_LABELS = {
    "wins": "Wins",
    "losses": "Losses",
    "payroll": "Payroll ($M)",
    "wins_per_10m": "Wins per $10M",
    "team_total_war": "Team WAR",
    "cost_per_war": "Cost per WAR ($M)",
    "surplus_value": "Surplus ($M)",
    "run_diff": "Run differential",
    "gini_salary": "Salary Gini",
    "pythag_wins": "Pythagorean wins",
    "pythag_gap": "Pythagorean gap",
    "dead_money_share": "Dead money %",
    "player_war": "WAR",
    "salary": "Salary ($M)",
    "window_phase": "Window phase",
}

CONTRACT_COLORS = {
    "surplus_value": "#3ee08f",
    "fair_value": "#6ecbff",
    "overpaid": "#f5c518",
    "dead_money": "#ff2d3a",
}

EFFICIENCY_COLORS = {
    "elite": "#3ee08f",
    "above_avg": "#6ecbff",
    "below_avg": "#f5c518",
    "low": "#ff2d3a",
}

# Same bins as pipeline.transform.build_metrics._efficiency_labels
_EFFICIENCY_BINS = [-float("inf"), 0.5, 1.0, 1.5, float("inf")]
_EFFICIENCY_LABELS = ["low", "below_avg", "above_avg", "elite"]

NAV_PAGES = (
    {
        "key": "overview",
        "index": "01",
        "label": "Overview",
        "kicker": "Command",
        "group": "League",
        "blurb": "Which teams bought the most wins per dollar this season?",
    },
    {
        "key": "deep_dive",
        "index": "02",
        "label": "Team Deep Dive",
        "kicker": "Franchise",
        "group": "League",
        "blurb": "Franchise trajectory: wins, payroll, WAR, and window phase.",
    },
    {
        "key": "compare",
        "index": "03",
        "label": "Compare Teams",
        "kicker": "League",
        "group": "League",
        "blurb": "Head-to-head trends across any metric and year range.",
    },
    {
        "key": "roster",
        "index": "04",
        "label": "Roster Lab",
        "kicker": "Roster",
        "group": "Roster",
        "blurb": "Player WAR versus salary, with contract classification.",
    },
    {
        "key": "contracts",
        "index": "05",
        "label": "Contract Watch",
        "kicker": "Contracts",
        "group": "Roster",
        "blurb": "Surplus value, overpays, and dead money — searchable by season.",
    },
    {
        "key": "frontier",
        "index": "06",
        "label": "Efficiency Frontier",
        "kicker": "Models",
        "group": "Models",
        "blurb": "Teams above the curve produce more wins per dollar than the baseline.",
    },
    {
        "key": "whatif",
        "index": "07",
        "label": "What-If Sim",
        "kicker": "Models",
        "group": "Models",
        "blurb": "Estimate win change from a payroll increase using league-wide regression.",
    },
    {
        "key": "models",
        "index": "08",
        "label": "Model Insights",
        "kicker": "Models",
        "group": "Models",
        "blurb": "Win-model accuracy, feature importance, and the largest misses.",
    },
)

FULL_PIPELINE = (
    "python3 -m pipeline.extract.pull_sources\n"
    "python3 -m pipeline.extract.pull_war\n"
    "python3 -m pipeline.extract.pull_mlb_stats\n"
    "python3 -m pipeline.extract.pull_sportsdataio\n"
    "python3 -m pipeline.transform.build_warehouse\n"
    "python3 -m pipeline.transform.build_metrics\n"
    "python3 -m models.train_win_model\n"
    "python3 -m models.cluster_teams"
)

_EMPTY_COPY = {
    "metrics": {
        "title": "No team metrics yet",
        "body": (
            "The dashboard reads CSVs from ARTIFACTS_URI/current/ when set, "
            "otherwise from artifacts/. Run the pipeline to generate team-season "
            "metrics, then refresh this page."
        ),
        "command": FULL_PIPELINE,
    },
    "players": {
        "title": "No player metrics yet",
        "body": "Roster and contract views need player_season_metrics.csv from the metrics step.",
        "command": "python3 -m pipeline.transform.build_metrics",
    },
    "frontier": {
        "title": "No frontier data yet",
        "body": "The payroll-vs-wins envelope is produced by the win-model training step.",
        "command": "python3 -m models.train_win_model",
    },
    "clusters": {
        "title": "No team archetypes yet",
        "body": "KMeans cluster labels are produced by the clustering step.",
        "command": "python3 -m models.cluster_teams",
    },
    "models": {
        "title": "No model artifacts yet",
        "body": "Train the win models to populate metrics, predictions, and feature importance.",
        "command": "python3 -m models.train_win_model",
    },
    "window": {
        "title": "No window-phase data yet",
        "body": "Franchise window phases are exported with the rest of the metric CSVs.",
        "command": "python3 -m pipeline.transform.build_metrics",
    },
    "season": {
        "title": "No teams for this season",
        "body": (
            "Try another year. Lahman payroll typically ends in 2016, so recent seasons "
            "may have standings without dollar metrics."
        ),
        "command": "",
    },
    "team": {
        "title": "No data for this team",
        "body": "Pick another franchise, or run the pipeline if artifacts are incomplete.",
        "command": "",
    },
    "compare": {
        "title": "Select at least two teams",
        "body": "Choose two or more franchises to compare wins, payroll, WAR, or surplus.",
        "command": "",
    },
    "generic": {
        "title": "Nothing to show",
        "body": "This view has no rows for the current filters.",
        "command": "",
    },
}


def nav_labels() -> list[str]:
    return [page["label"] for page in NAV_PAGES]


def nav_page(label: str) -> dict[str, str]:
    for page in NAV_PAGES:
        if page["label"] == label:
            return page
    return NAV_PAGES[0]


def nav_groups() -> list[tuple[str, list[dict[str, str]]]]:
    """Group nav pages for product-style sidebar sections."""
    grouped: dict[str, list[dict[str, str]]] = {}
    order: list[str] = []
    for page in NAV_PAGES:
        group = str(page.get("group") or "More")
        if group not in grouped:
            grouped[group] = []
            order.append(group)
        grouped[group].append(page)
    return [(name, grouped[name]) for name in order]


def delta_tone(delta: str | None) -> str:
    """Return pos / neg / neu for a formatted delta string."""
    if delta is None:
        return "neu"
    text = str(delta).strip()
    if not text or text in {"—", "-", "0", "$0M", "+0", "+$0M"}:
        return "neu"
    if text.startswith("-"):
        return "neg"
    return "pos"


def kpi_cards_html(cards: list[dict[str, Any]]) -> str:
    """Dense command-center KPI strip. Values are escaped."""
    cells: list[str] = []
    for card in cards:
        label = html.escape(str(card.get("label") or ""))
        value = html.escape(str(card.get("value") if card.get("value") is not None else "—"))
        delta = card.get("delta")
        delta_html = ""
        if delta:
            tone = delta_tone(str(delta))
            delta_html = f'<div class="kpi-delta {tone}">{html.escape(str(delta))}</div>'
        cells.append(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f"{delta_html}"
            f"</div>"
        )
    return f'<div class="kpi-grid">{"".join(cells)}</div>'


def leaderboard_html(
    df: pd.DataFrame,
    *,
    name_col: str = "team_name",
    value_col: str,
    value_format: str = "{:.1f}",
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Compact ranked list with proportional bars for Overview leader panels."""
    parsed: list[tuple[int, str, str, float]] = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        data = row._asdict() if hasattr(row, "_asdict") else dict(zip(df.columns, row))
        name = html.escape(str(data.get(name_col, "—")))
        raw = data.get(value_col)
        numeric = 0.0
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            formatted = "—"
        else:
            try:
                numeric = float(raw)
                formatted = f"{prefix}{value_format.format(numeric)}{suffix}"
            except (TypeError, ValueError):
                formatted = str(raw)
        parsed.append((idx, name, formatted, numeric))
    peak = max((abs(item[3]) for item in parsed), default=1.0) or 1.0
    rows: list[str] = []
    for idx, name, formatted, numeric in parsed:
        width = max(6, min(100, int(round(100 * abs(numeric) / peak))))
        tone = " neg" if numeric < 0 else ""
        rows.append(
            f'<li class="lb-row{tone}">'
            f'<span class="lb-rank">{idx:02d}</span>'
            f'<span class="lb-name">{name}</span>'
            f'<span class="lb-bar"><i style="width:{width}%"></i></span>'
            f'<span class="lb-stat">{html.escape(formatted)}</span>'
            f"</li>"
        )
    return f'<ol class="leaderboard">{"".join(rows)}</ol>'


def masthead_html(label: str, extra_caption: str | None = None) -> str:
    """Numbered page masthead — used instead of ``st.title``."""
    meta = nav_page(label)
    extra = (
        f'<p class="war-note">{html.escape(extra_caption)}</p>' if extra_caption else ""
    )
    index = html.escape(str(meta.get("index") or ""))
    kicker = html.escape(str(meta.get("kicker") or "Ops"))
    return (
        f'<div class="masthead">'
        f'<div class="kicker">{index} · {kicker}</div>'
        f"<h1>{html.escape(meta['label'])}</h1>"
        f'<p class="blurb">{html.escape(meta["blurb"])}</p>'
        f"{extra}"
        f"</div>"
    )


def app_frame_html(
    *,
    seasons: str,
    artifacts: str,
    source: str,
    page: str,
) -> str:
    """Top command bar — seasons / artifacts / source / active desk."""
    meta = nav_page(page)
    index = html.escape(str(meta.get("index") or ""))
    return (
        '<div class="app-frame">'
        '<div class="frame-brand">EE<span>OPS</span></div>'
        '<div class="frame-meta">'
        f"<span>Seasons <b>{html.escape(seasons)}</b></span><i></i>"
        f"<span>Artifacts <b>{html.escape(artifacts)}</b></span><i></i>"
        f"<span>Source <b>{html.escape(source)}</b></span><i></i>"
        f"<span>Desk <b>{index} {html.escape(page)}</b></span>"
        "</div></div>"
    )


def dossier_html(
    *,
    team: str,
    year: int | str | None,
    wins: Any = None,
    losses: Any = None,
    phase: str = "",
) -> str:
    """Franchise identity block for Team Deep Dive."""
    w = "—" if _is_missing(wins) else str(int(wins))
    l = "—" if _is_missing(losses) else str(int(losses))
    year_label = "—" if year is None else str(year)
    badge = ""
    if phase and phase.lower() not in {"nan", "none", "—"}:
        badge = f'<span class="phase-badge">{html.escape(phase.title())}</span>'
    return (
        f'<div class="dossier">'
        f'<div class="kicker">Club dossier · {html.escape(year_label)}</div>'
        f'<div class="name">{html.escape(team)}</div>'
        f'<div class="line"><span class="wl">{html.escape(w)}<span class="l">–{html.escape(l)}</span></span>{badge}</div>'
        f"</div>"
    )


def scoreboard_html(items: list[tuple[str, Any]]) -> str:
    """Dense stat strip used instead of a row of ``st.metric`` widgets."""
    cells: list[str] = []
    for label, value in items:
        display = "—" if value is None or value == "" else str(value)
        cells.append(
            f'<div class="sb-cell"><span class="sb-k">{html.escape(str(label))}</span>'
            f'<span class="sb-v">{html.escape(display)}</span></div>'
        )
    n = len(items) or 1
    return f'<div class="scoreboard n{n}">{"".join(cells)}</div>'


def metric_label(column: str) -> str:
    return METRIC_LABELS.get(column, column.replace("_", " ").title())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def format_money_millions(value: Any, *, decimals: int = 0, na: str = "—") -> str:
    """Format a raw dollar amount as $XM / $X.XM. Negatives render as -$XM."""
    if _is_missing(value):
        return na
    amount = float(value) / 1_000_000
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.{decimals}f}M"


def format_war(value: Any, *, decimals: int = 1, na: str = "—") -> str:
    if _is_missing(value):
        return na
    return f"{float(value):,.{decimals}f}"


def format_signed_int(value: Any, *, na: str = "—") -> str:
    if _is_missing(value):
        return na
    return f"{int(value):+d}"


def format_ratio(value: Any, *, decimals: int = 2, na: str = "—") -> str:
    if _is_missing(value):
        return na
    return f"{float(value):.{decimals}f}"


def scale_money_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert payroll/salary columns from raw $ to $M and dead-money share to %."""
    out = df.copy()
    for col in MONEY_COLS_TO_MILLIONS:
        if col in out.columns:
            out[col] = out[col] / 1_000_000
    if "dead_money_share" in out.columns:
        out["dead_money_share"] = out["dead_money_share"] * 100
    return out


def add_payroll_millions(df: pd.DataFrame, source: str = "payroll") -> pd.DataFrame:
    """Add a payroll_m column for charts without mutating the source dollars."""
    out = df.copy()
    if source in out.columns and "payroll_m" not in out.columns:
        out["payroll_m"] = out[source] / 1_000_000
    return out


def years_from_frame(df: pd.DataFrame | None, column: str = "year_id") -> list[int]:
    if df is None or df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(int).unique().tolist())


def teams_from_frame(df: pd.DataFrame | None, column: str = "team_name") -> list[str]:
    if df is None or df.empty or column not in df.columns:
        return []
    return sorted(df[column].dropna().astype(str).unique().tolist())


def slider_bounds(years: list[int], current_year: int) -> tuple[int, int]:
    if not years:
        return current_year, current_year
    return int(years[0]), max(int(years[-1]), int(current_year))


def filter_season(
    metrics: pd.DataFrame,
    year: int,
    league: str = "All",
) -> pd.DataFrame:
    season = metrics[metrics["year_id"] == year].copy()
    if league != "All" and "league_id" in season.columns:
        season = season[season["league_id"] == league]
    return season


def efficiency_sort_columns(df: pd.DataFrame) -> list[str]:
    """Preferred sort keys so the first row is the most efficient team."""
    cols: list[str] = []
    if "surplus_value" in df.columns and df["surplus_value"].notna().any():
        cols.append("surplus_value")
    if "wins_per_10m" in df.columns and df["wins_per_10m"].notna().any():
        cols.append("wins_per_10m")
    if "wins" in df.columns:
        cols.append("wins")
    return cols


def rank_by_efficiency(season: pd.DataFrame) -> pd.DataFrame:
    """Sort a season table by efficiency and add a 1-based rank column."""
    df = season.copy()
    sort_cols = efficiency_sort_columns(df)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    df = df.reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def apply_efficiency_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add efficiency_label from wins_per_10m when the export column is absent."""
    out = df.copy()
    if "efficiency_label" in out.columns or "wins_per_10m" not in out.columns:
        return out
    out["efficiency_label"] = pd.cut(
        out["wins_per_10m"],
        bins=_EFFICIENCY_BINS,
        labels=_EFFICIENCY_LABELS,
    )
    return out


def _team_at_extreme(
    season: pd.DataFrame,
    column: str,
    *,
    least: bool = False,
    min_war: float | None = None,
) -> pd.Series | None:
    if column not in season.columns:
        return None
    valid = season.dropna(subset=[column])
    if min_war is not None and "team_total_war" in valid.columns:
        filtered = valid[valid["team_total_war"].fillna(0) >= min_war]
        if not filtered.empty:
            valid = filtered
    if valid.empty:
        return None
    idx = valid[column].idxmin() if least else valid[column].idxmax()
    return valid.loc[idx]


def overview_leaders(season: pd.DataFrame) -> dict[str, Any]:
    """Named leaders used by the Overview KPI row."""
    result: dict[str, Any] = {
        "n_teams": int(len(season)),
        "best_surplus_team": None,
        "best_surplus": None,
        "worst_surplus_team": None,
        "worst_surplus": None,
        "best_cpw_team": None,
        "best_cost_per_war": None,
        "best_wp10_team": None,
        "best_wins_per_10m": None,
        "median_payroll": None,
        "n_positive_surplus": 0,
        "avg_team_war": None,
        "has_dollar_metrics": False,
    }
    if season.empty:
        return result

    if "payroll" in season.columns and season["payroll"].notna().any():
        result["median_payroll"] = float(season["payroll"].median())
        result["has_dollar_metrics"] = True

    if "team_total_war" in season.columns and season["team_total_war"].notna().any():
        result["avg_team_war"] = float(season["team_total_war"].mean())

    best_sv = _team_at_extreme(season, "surplus_value")
    if best_sv is not None:
        result["best_surplus_team"] = best_sv.get("team_name")
        result["best_surplus"] = best_sv["surplus_value"]
        result["has_dollar_metrics"] = True
        valid_sv = season.dropna(subset=["surplus_value"])
        result["n_positive_surplus"] = int((valid_sv["surplus_value"] > 0).sum())
        worst_sv = _team_at_extreme(season, "surplus_value", least=True)
        if worst_sv is not None:
            result["worst_surplus_team"] = worst_sv.get("team_name")
            result["worst_surplus"] = worst_sv["surplus_value"]

    best_cpw = _team_at_extreme(season, "cost_per_war", least=True, min_war=5.0)
    if best_cpw is not None:
        result["best_cpw_team"] = best_cpw.get("team_name")
        result["best_cost_per_war"] = best_cpw["cost_per_war"]
        result["has_dollar_metrics"] = True

    best_wp10 = _team_at_extreme(season, "wins_per_10m")
    if best_wp10 is not None:
        result["best_wp10_team"] = best_wp10.get("team_name")
        result["best_wins_per_10m"] = best_wp10["wins_per_10m"]
        result["has_dollar_metrics"] = True

    return result


def overview_kpi_payload(season: pd.DataFrame) -> list[dict[str, str | None]]:
    """Display-ready KPI cards: label, value, optional delta."""
    leaders = overview_leaders(season)
    if season.empty:
        return [
            {"label": "Teams", "value": "0", "delta": None},
            {"label": "Most surplus", "value": "—", "delta": None},
            {"label": "Lowest $/WAR", "value": "—", "delta": None},
            {"label": "Best W/$10M", "value": "—", "delta": None},
            {"label": "Teams in surplus", "value": "—", "delta": None},
            {"label": "Median payroll", "value": "—", "delta": None},
        ]

    surplus_delta = (
        format_money_millions(leaders["best_surplus"], decimals=0)
        if leaders["best_surplus"] is not None
        else None
    )
    cpw_delta = (
        format_money_millions(leaders["best_cost_per_war"], decimals=1)
        if leaders["best_cost_per_war"] is not None
        else None
    )
    wp10_delta = (
        format_ratio(leaders["best_wins_per_10m"])
        if leaders["best_wins_per_10m"] is not None
        else None
    )
    surplus_count = (
        f"{leaders['n_positive_surplus']} / {leaders['n_teams']}"
        if leaders["has_dollar_metrics"]
        else "—"
    )
    return [
        {"label": "Teams", "value": str(leaders["n_teams"]), "delta": None},
        {
            "label": "Most surplus",
            "value": leaders["best_surplus_team"] or "—",
            "delta": surplus_delta,
        },
        {
            "label": "Lowest $/WAR",
            "value": leaders["best_cpw_team"] or "—",
            "delta": cpw_delta,
        },
        {
            "label": "Best W/$10M",
            "value": leaders["best_wp10_team"] or "—",
            "delta": wp10_delta,
        },
        {"label": "Teams in surplus", "value": surplus_count, "delta": None},
        {
            "label": "Median payroll",
            "value": format_money_millions(leaders["median_payroll"], decimals=0),
            "delta": None,
        },
    ]


def top_n_by(
    season: pd.DataFrame,
    column: str,
    n: int = 5,
    *,
    ascending: bool = False,
    extra_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    if column not in season.columns:
        return season.head(0)
    valid = season.dropna(subset=[column])
    keep = [c for c in ("team_name", column, *extra_cols) if c in valid.columns]
    ranked = valid.sort_values(column, ascending=ascending, na_position="last")
    return ranked[keep].head(n).reset_index(drop=True)


def salary_coverage_note(year: int | None) -> str | None:
    if year is None:
        return None
    if int(year) > SALARY_DATA_LAST_YEAR:
        return (
            f"Lahman salary and payroll typically end in {SALARY_DATA_LAST_YEAR}. "
            "Dollar metrics (payroll, surplus, $/WAR) may be blank for this season."
        )
    return None


def artifact_status(files: dict[str, Path | None]) -> dict[str, Any]:
    present = {
        key: path is not None and Path(path).exists() for key, path in files.items()
    }
    missing = [key for key, ok in present.items() if not ok]
    return {
        "present": present,
        "n_present": sum(present.values()),
        "n_total": len(present),
        "missing": missing,
        "ready": bool(present.get("metrics")),
    }


def empty_state_copy(kind: str) -> dict[str, str]:
    return dict(_EMPTY_COPY.get(kind, _EMPTY_COPY["generic"]))


def year_span_label(years: list[int]) -> str:
    if not years:
        return "No seasons loaded"
    if years[0] == years[-1]:
        return str(years[0])
    return f"{years[0]}–{years[-1]}"
