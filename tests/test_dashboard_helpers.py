from __future__ import annotations

import pytest

import ast
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from datetime import date

from dashboard.helpers import (
    PRIOR_SEASON_TABLE_NOTE,
    add_payroll_millions,
    apply_efficiency_labels,
    artifact_status,
    blank_unknown_salary,
    clamp_season_for_page,
    data_slider_max,
    empty_state_copy,
    filter_contract_watch_rows,
    filter_season,
    format_money_millions,
    format_ratio,
    format_signed_int,
    format_war,
    is_prior_only_publish,
    max_season_from_cards,
    max_season_from_frame,
    metric_label,
    nav_labels,
    nav_page,
    overview_kpi_payload,
    overview_leaders,
    rank_by_efficiency,
    resolve_active_year,
    salary_coverage_note,
    scale_money_columns,
    seasons_from_manifest,
    slider_bounds,
    teams_from_frame,
    top_n_by,
    year_span_from_frame,
    year_span_label,
    years_from_frame,
)
from src.baseball_analytics.dashboard_helpers import (
    apply_layout_and_render_chart,
    compute_slider_max,
)

pytestmark = pytest.mark.unit


def _season() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "team_name": ["Rays", "Yankees", "A's", "Red Sox"],
            "league_id": ["AL", "AL", "AL", "AL"],
            "year_id": [2015, 2015, 2015, 2015],
            "wins": [80, 87, 68, 78],
            "payroll": [70_000_000, 210_000_000, 80_000_000, 180_000_000],
            "team_total_war": [28.0, 35.0, 18.0, 30.0],
            "surplus_value": [40_000_000, -30_000_000, 10_000_000, -20_000_000],
            "cost_per_war": [2_500_000, 6_000_000, 4_400_000, 6_000_000],
            "wins_per_10m": [1.14, 0.41, 0.85, 0.43],
        }
    )


def test_compute_slider_max_uses_current_year_when_no_data() -> None:
    assert compute_slider_max([], 2026) == 2026


def test_compute_slider_max_uses_latest_metric_year_when_greater() -> None:
    assert compute_slider_max([2018, 2019, 2020], 2017) == 2020


def test_compute_slider_max_uses_current_year_when_greater() -> None:
    assert compute_slider_max([2018, 2019, 2020], 2026) == 2026


def test_apply_layout_and_render_chart_applies_layout_then_renders() -> None:
    events: list[str] = []

    class DummyFigure:
        def __init__(self) -> None:
            self.last_layout_kwargs: dict[str, int] | None = None

        def update_layout(self, **kwargs: int) -> None:
            events.append("update_layout")
            self.last_layout_kwargs = kwargs

    fig = DummyFigure()

    def fake_apply_layout(figure: DummyFigure) -> None:
        events.append("apply_layout")
        assert figure is fig

    plotly_chart = Mock()
    apply_layout_and_render_chart(
        fig,
        apply_layout=fake_apply_layout,
        plotly_chart=plotly_chart,
        height=460,
    )

    assert events == ["apply_layout", "update_layout"]
    assert fig.last_layout_kwargs == {"height": 460}
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_nav_has_eight_product_sections() -> None:
    labels = nav_labels()
    assert labels == [
        "Overview",
        "Team Deep Dive",
        "Compare Teams",
        "Roster Lab",
        "Contract Watch",
        "Efficiency Frontier",
        "What-If Sim",
        "Model Insights",
    ]
    assert nav_page("Roster Lab")["kicker"] == "Roster"
    assert nav_page("Missing")["label"] == "Overview"

    from dashboard.helpers import kpi_cards_html, nav_groups

    groups = dict(nav_groups())
    assert list(groups) == ["League", "Roster", "Models"]
    assert [p["label"] for p in groups["League"]] == ["Overview", "Team Deep Dive", "Compare Teams"]
    html = kpi_cards_html([{"label": "Teams", "value": "30", "delta": None}])
    assert "Teams" in html and "30" in html


def test_format_money_and_war() -> None:
    assert format_money_millions(98_400_000) == "$98M"
    assert format_money_millions(2_500_000, decimals=1) == "$2.5M"
    assert format_money_millions(-30_400_000) == "-$30M"
    assert format_money_millions(-2_500_000, decimals=1) == "-$2.5M"
    assert format_money_millions(None) == "—"
    assert format_money_millions(float("nan")) == "—"
    assert format_war(32.46) == "32.5"
    assert format_war(None) == "—"
    assert format_signed_int(42) == "+42"
    assert format_signed_int(-7) == "-7"
    assert format_ratio(1.141) == "1.14"


def test_scale_money_columns_and_dead_money() -> None:
    df = pd.DataFrame(
        {
            "payroll": [100_000_000],
            "surplus_value": [8_000_000],
            "salary": [2_000_000],
            "dead_money_share": [0.12],
            "wins": [90],
        }
    )
    scaled = scale_money_columns(df)
    assert scaled["payroll"].iloc[0] == 100.0
    assert scaled["surplus_value"].iloc[0] == 8.0
    assert scaled["salary"].iloc[0] == 2.0
    assert abs(scaled["dead_money_share"].iloc[0] - 12.0) < 1e-9
    assert scaled["wins"].iloc[0] == 90
    assert df["payroll"].iloc[0] == 100_000_000


def test_add_payroll_millions() -> None:
    df = pd.DataFrame({"payroll": [50_000_000], "wins": [81]})
    chart = add_payroll_millions(df)
    assert chart["payroll_m"].iloc[0] == 50.0
    assert chart["payroll"].iloc[0] == 50_000_000


def test_years_teams_and_slider_bounds() -> None:
    df = pd.DataFrame({"year_id": [2014, 2016, 2015], "team_name": ["A", "B", "A"]})
    assert years_from_frame(df) == [2014, 2015, 2016]
    assert teams_from_frame(df) == ["A", "B"]
    assert years_from_frame(None) == []
    assert teams_from_frame(pd.DataFrame()) == []
    assert slider_bounds([1990, 2016], 2026) == (1990, 2026)
    assert slider_bounds([], 2026) == (2026, 2026)


def test_filter_season_by_league() -> None:
    df = pd.DataFrame(
        {
            "year_id": [2015, 2015, 2014],
            "league_id": ["AL", "NL", "AL"],
            "team_name": ["Yankees", "Mets", "Royals"],
        }
    )
    al = filter_season(df, 2015, "AL")
    assert list(al["team_name"]) == ["Yankees"]
    assert len(filter_season(df, 2015, "All")) == 2


def test_rank_by_efficiency_prefers_surplus() -> None:
    ranked = rank_by_efficiency(_season())
    assert list(ranked["team_name"]) == ["Rays", "A's", "Red Sox", "Yankees"]
    assert list(ranked["rank"]) == [1, 2, 3, 4]


def test_rank_by_efficiency_falls_back_to_wins() -> None:
    df = pd.DataFrame({"team_name": ["A", "B"], "wins": [70, 95]})
    ranked = rank_by_efficiency(df)
    assert list(ranked["team_name"]) == ["B", "A"]


def test_apply_efficiency_labels_matches_pipeline_bins() -> None:
    df = pd.DataFrame({"wins_per_10m": [0.2, 0.7, 1.2, 2.0]})
    labeled = apply_efficiency_labels(df)
    assert list(labeled["efficiency_label"].astype(str)) == [
        "low",
        "below_avg",
        "above_avg",
        "elite",
    ]
    already = pd.DataFrame({"wins_per_10m": [2.0], "efficiency_label": ["custom"]})
    assert apply_efficiency_labels(already)["efficiency_label"].iloc[0] == "custom"


def test_overview_leaders_and_kpis() -> None:
    leaders = overview_leaders(_season())
    assert leaders["n_teams"] == 4
    assert leaders["best_surplus_team"] == "Rays"
    assert leaders["worst_surplus_team"] == "Yankees"
    assert leaders["best_cpw_team"] == "Rays"
    assert leaders["best_wp10_team"] == "Rays"
    assert leaders["n_positive_surplus"] == 2
    assert leaders["has_dollar_metrics"] is True

    cards = overview_kpi_payload(_season())
    assert cards[1]["value"] == "Rays"
    assert cards[1]["delta"] == "$40M"
    assert cards[4]["value"] == "2 / 4"
    assert cards[5]["value"] == "$130M"

    empty = overview_kpi_payload(pd.DataFrame())
    assert empty[0]["value"] == "0"
    assert empty[1]["value"] == "—"


def test_top_n_by_and_metric_label() -> None:
    top = top_n_by(_season(), "surplus_value", n=2, extra_cols=("wins",))
    assert list(top["team_name"]) == ["Rays", "A's"]
    bottom = top_n_by(_season(), "surplus_value", n=1, ascending=True)
    assert bottom["team_name"].iloc[0] == "Yankees"
    assert metric_label("surplus_value") == "Surplus ($M)"
    assert metric_label("unknown_col") == "Unknown Col"


def test_salary_coverage_note() -> None:
    assert salary_coverage_note(2016) is None
    note = salary_coverage_note(2024)
    assert note is not None
    assert "2016" in note
    assert salary_coverage_note(None) is None


def test_filter_contract_watch_keeps_missing_and_zero_salary() -> None:
    players = pd.DataFrame(
        {
            "name_full": ["Overlay 2026", "Zero Cap", "Paid Veteran"],
            "year_id": [2026, 2026, 2016],
            "team_name": ["Padres", "Padres", "Padres"],
            "salary": [float("nan"), 0, 8_000_000],
            "surplus_value": [float("nan"), 0, 1_000_000],
        }
    )
    kept = filter_contract_watch_rows(players, year=2026, team="Padres")
    assert list(kept["name_full"]) == ["Overlay 2026", "Zero Cap"]
    assert kept["salary"].isna().iloc[0]
    assert kept["salary"].iloc[1] == 0

    searched = filter_contract_watch_rows(players, name_search="overlay")
    assert list(searched["name_full"]) == ["Overlay 2026"]


def test_blank_unknown_salary_leaves_paid_rows() -> None:
    df = pd.DataFrame({"salary": [float("nan"), 0, 2.5], "name_full": ["A", "B", "C"]})
    blanked = blank_unknown_salary(df)
    assert pd.isna(blanked["salary"].iloc[0])
    assert pd.isna(blanked["salary"].iloc[1])
    assert blanked["salary"].iloc[2] == 2.5
    assert list(blanked["name_full"]) == ["A", "B", "C"]


def test_prior_only_banner_condition() -> None:
    assert is_prior_only_publish(
        current_season_missing=True,
        max_season=2024,
        active_year=2026,
    )
    assert is_prior_only_publish(
        current_season_missing=False,
        max_season=2024,
        active_year=2026,
    )
    assert is_prior_only_publish(
        current_season_missing=False,
        selected_season=2024,
        max_season=2026,
        active_year=2026,
    )
    assert not is_prior_only_publish(
        current_season_missing=False,
        selected_season=2026,
        max_season=2026,
        active_year=2026,
    )
    assert not is_prior_only_publish(
        current_season_missing=True,
        max_season=2024,
        active_year=2026,
        live_feed=False,
    )
    assert not is_prior_only_publish(
        current_season_missing=False,
        max_season=None,
        active_year=2026,
    )
    assert is_prior_only_publish(
        current_season_missing=False,
        max_season=2026,
        seasons_present=[2023, 2024],
        active_year=2026,
    )
    assert not is_prior_only_publish(
        current_season_missing=False,
        seasons_present=[2024, 2025, 2026],
        active_year=2026,
    )
    assert PRIOR_SEASON_TABLE_NOTE == "This table is not the current season yet."
    assert seasons_from_manifest({"seasons_present": [2026, 2024, "x"]}) == [2024, 2026]
    assert seasons_from_manifest(None) == []


def test_data_slider_max_uses_published_years_only() -> None:
    assert data_slider_max([], 2026) == 2026
    assert data_slider_max([1990, 2016], 2026) == 2016
    assert year_span_from_frame(pd.DataFrame({"year_id": [2010, 2016]})) == (2010, 2016)
    assert year_span_from_frame(pd.DataFrame()) is None


def test_clamp_season_for_page_does_not_write_player_only_year() -> None:
    display, write_shared = clamp_season_for_page(2026, [1990, 2015, 2016])
    assert display == 2016
    assert write_shared is False
    display, write_shared = clamp_season_for_page(2015, [1990, 2015, 2016])
    assert display == 2015
    assert write_shared is True
    assert clamp_season_for_page(2026, []) == (None, False)


def test_resolve_active_year_prefers_manifest_then_as_of() -> None:
    assert resolve_active_year(manifest={"active_season": 2026}, today=date(2024, 1, 1)) == 2026
    assert resolve_active_year(manifest={"as_of_date": "2026-08-23"}, today=date(2024, 1, 1)) == 2026
    assert resolve_active_year(as_of="2025-07-01", today=date(2024, 1, 1)) == 2025
    assert resolve_active_year(today=date(2026, 8, 24)) == 2026


def test_max_season_from_cards_and_frame() -> None:
    assert max_season_from_cards([{"season": 2024}, {"season": 2025}]) == 2025
    assert max_season_from_cards([{"as_of_date": "2026-08-23"}]) is None
    assert max_season_from_cards([]) is None
    df = pd.DataFrame({"year_id": [2016, 2024, 2024]})
    assert max_season_from_frame(df) == 2024
    assert max_season_from_frame(pd.DataFrame()) is None


def test_artifact_status_and_empty_copy(tmp_path: Path) -> None:
    present = tmp_path / "team_onfield_contract_metrics.csv"
    present.write_text("year_id\n2015\n")
    files = {
        "metrics": present,
        "players": tmp_path / "missing.csv",
    }
    status = artifact_status(files)
    assert status["n_present"] == 1
    assert status["n_total"] == 2
    assert status["ready"] is True
    assert status["missing"] == ["players"]

    none_status = artifact_status({"metrics": None, "players": present})
    assert none_status["n_present"] == 1
    assert none_status["missing"] == ["metrics"]

    copy = empty_state_copy("players")
    assert "player_season_metrics.csv" in copy["body"]
    assert empty_state_copy("not-a-kind")["title"] == "Nothing to show"


def test_year_span_label() -> None:
    assert year_span_label([]) == "No seasons loaded"
    assert year_span_label([2015]) == "2015"
    assert year_span_label([1990, 2016]) == "1990–2016"


def test_app_bootstraps_sys_path_before_package_imports() -> None:
    """Streamlit adds dashboard/ (not the repo root) to sys.path first."""
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    tree = ast.parse(app_path.read_text())
    saw_root_bootstrap = False
    package_imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_ROOT" in targets:
                saw_root_bootstrap = True
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "dashboard" or node.module.startswith(("dashboard.", "src.")):
                package_imports.append(node.module)
                assert saw_root_bootstrap, f"{node.module} imported before _ROOT sys.path bootstrap"
    assert saw_root_bootstrap
    assert "dashboard.helpers" in package_imports
    assert "src.baseball_analytics.dashboard_utils" in package_imports
    assert package_imports[0].startswith("src.") or package_imports[0].startswith("dashboard.")
