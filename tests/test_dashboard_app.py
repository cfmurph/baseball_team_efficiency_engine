from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from dashboard.helpers import (
    CONTRACT_COLORS,
    add_payroll_millions,
    empty_state_copy,
    format_money_millions,
    format_signed_int,
    format_war,
    nav_page,
    scale_money_columns,
    teams_from_frame,
    years_from_frame,
)
from src.baseball_analytics.dashboard_utils import player_id_columns_for_duplicate_names


APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def _read_app_tree() -> ast.Module:
    return ast.parse(APP_PATH.read_text())


def _load_app_symbols(
    *,
    functions: tuple[str, ...] = (),
    assignments: tuple[str, ...] = (),
    globals_dict: dict | None = None,
) -> dict:
    tree = _read_app_tree()
    selected_nodes: list[ast.stmt] = []
    wanted_funcs = set(functions)
    wanted_assignments = set(assignments)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
            selected_nodes.append(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted_assignments:
                    selected_nodes.append(node)
                    break

    module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {}
    if globals_dict:
        namespace.update(globals_dict)
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


def _slider_max_expr() -> ast.expr:
    tree = _read_app_tree()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_slider_max":
                    return node.value
    raise AssertionError("Could not find _slider_max assignment in dashboard/app.py")


class _DummyElement:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def metric(self, *args, **kwargs) -> None:
        return None


class _FakeColumnConfig:
    @staticmethod
    def TextColumn(label: str, *args, **kwargs):
        return {"kind": "text", "label": label}

    @staticmethod
    def NumberColumn(label: str, *args, **kwargs):
        return {"kind": "number", "label": label}

    @staticmethod
    def CheckboxColumn(label: str, *args, **kwargs):
        return {"kind": "checkbox", "label": label}


class _FakeStreamlit:
    def __init__(self, *, selectbox_values: dict | None = None):
        self.selectbox_values = selectbox_values or {}
        self.captions: list[str] = []
        self.session_state: dict = {}
        self.plotly_chart_calls: list[tuple[object, bool]] = []
        self.column_config = _FakeColumnConfig()

    def title(self, *args, **kwargs) -> None:
        return None

    def markdown(self, *args, **kwargs) -> None:
        return None

    def metric(self, *args, **kwargs) -> None:
        return None

    def caption(self, text: str, *args, **kwargs) -> None:
        self.captions.append(str(text))

    def markdown(self, *args, **kwargs) -> None:
        return None

    def warning(self, *args, **kwargs) -> None:
        return None

    def info(self, *args, **kwargs) -> None:
        return None

    def divider(self, *args, **kwargs) -> None:
        return None

    def subheader(self, *args, **kwargs) -> None:
        return None

    def columns(self, spec, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec)
        return tuple(_DummyElement() for _ in range(n))

    def tabs(self, labels, *args, **kwargs):
        return tuple(_DummyElement() for _ in labels)

    def expander(self, *args, **kwargs):
        return _DummyElement()

    def button(self, *args, **kwargs) -> bool:
        return False

    def selectbox(self, label, options, index=0, key=None, **kwargs):
        if key in self.selectbox_values:
            return self.selectbox_values[key]
        opts = list(options)
        if not opts:
            return None
        if index is None:
            return opts[0]
        return opts[min(max(index, 0), len(opts) - 1)]

    def text_input(self, label, value="", key=None, **kwargs):
        return value if isinstance(value, str) else ""

    def plotly_chart(self, fig, use_container_width=False, **kwargs) -> None:
        self.plotly_chart_calls.append((fig, bool(use_container_width)))


class _FakeFigure:
    def __init__(self):
        self.layouts: list[dict] = []

    def update_layout(self, **kwargs) -> None:
        self.layouts.append(dict(kwargs))

    def add_scatter(self, *args, **kwargs) -> None:
        return None

    def add_hline(self, *args, **kwargs) -> None:
        return None

    def update_yaxes(self, *args, **kwargs) -> None:
        return None

    def update_traces(self, *args, **kwargs) -> None:
        return None

    def add_hline(self, *args, **kwargs) -> None:
        return None


def test_nav_pages_resolve_to_defined_page_functions():
    tree = _read_app_tree()
    func_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    aliases: dict[str, str] = {}
    pages = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_PAGES" and isinstance(node.value, ast.Dict):
                pages = node.value
            elif isinstance(target, ast.Name) and isinstance(node.value, ast.Name):
                aliases[target.id] = node.value.id
    assert pages is not None, "Could not find _PAGES routing dict"
    for key, value in zip(pages.keys, pages.values):
        assert isinstance(key, ast.Constant)
        assert isinstance(value, ast.Name)
        resolved = aliases.get(value.id, value.id)
        assert resolved in func_names, f"{key.value} maps to undefined {value.id}"


def test_slider_max_expression_handles_empty_and_non_empty_years():
    expr = _slider_max_expr()
    compiled = compile(ast.Expression(expr), str(APP_PATH), "eval")

    from src.baseball_analytics.dashboard_helpers import compute_slider_max

    empty_result = eval(
        compiled,
        {"all_years": [], "_current_year": 2026, "compute_slider_max": compute_slider_max},
    )
    non_empty_result = eval(
        compiled,
        {"all_years": [2018, 2024], "_current_year": 2020, "compute_slider_max": compute_slider_max},
    )

    assert empty_result == 2026
    assert non_empty_result == 2024


def test_chart_applies_layout_and_renders_once():
    st = _FakeStreamlit()
    namespace = _load_app_symbols(
        functions=("_apply_layout", "_chart"),
        assignments=("_PLOTLY_LAYOUT",),
        globals_dict={"st": st},
    )
    chart = namespace["_chart"]
    fig = _FakeFigure()

    chart(fig, height=333)

    assert len(fig.layouts) >= 2
    assert fig.layouts[-1]["height"] == 333
    assert len(st.plotly_chart_calls) == 1
    rendered_fig, rendered_wide = st.plotly_chart_calls[0]
    assert rendered_fig is fig
    assert rendered_wide is True


def test_player_explorer_shows_player_id_when_name_collides():
    st = _FakeStreamlit(
        selectbox_values={
            "pe_year": 2024,
            "pe_team": "All Teams",
            "pe_type": "All Types",
            "pe_sort": "player_war",
        }
    )
    players = pd.DataFrame(
        [
            {
                "player_id": "alpha-1",
                "name_full": "Alex Gonzalez",
                "year_id": 2024,
                "team_name": "A",
                "player_type": "H",
                "pa": 100,
                "hr": 5,
                "bb": 10,
                "woba": 0.31,
                "batting_war": 1.2,
                "ip": 0.0,
                "era": 0.0,
                "fip": 0.0,
                "pitching_war": 0.0,
                "player_war": 1.2,
                "salary": 1_000_000,
                "surplus_value": 500_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "alpha-2",
                "name_full": "Alex Gonzalez",
                "year_id": 2024,
                "team_name": "B",
                "player_type": "P",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": 0.0,
                "batting_war": 0.0,
                "ip": 50.0,
                "era": 3.2,
                "fip": 3.5,
                "pitching_war": 1.8,
                "player_war": 1.8,
                "salary": 1_500_000,
                "surplus_value": 700_000,
                "contract_label": "surplus_value",
            },
        ]
    )

    captured_tables: list[pd.DataFrame] = []

    def _fake_load(key: str):
        if key == "players":
            return players
        if key == "sr_players":
            return None
        return None

    fake_px = SimpleNamespace(
        scatter=lambda *args, **kwargs: _FakeFigure(),
    )
    from dashboard.helpers import (
        empty_state_copy,
        metric_label,
        salary_coverage_note,
        scale_money_columns,
        teams_from_frame,
        years_from_frame,
    )
    from src.baseball_analytics.dashboard_utils import player_id_columns_for_duplicate_names

    namespace = _load_app_symbols(
        functions=("page_player_explorer", "_empty", "_salary_note", "_page_header"),
        assignments=("_SCATTER_MARKER",),
        globals_dict={
            "pd": pd,
            "st": st,
            "px": fake_px,
            "_load": _fake_load,
            "_scale_payroll": lambda df: df,
            "_PLAYER_COL_CFG": {},
            "_show_table": lambda df, *args, **kwargs: captured_tables.append(df.copy()),
            "_empty": lambda *args, **kwargs: None,
            "_page_header": lambda *args, **kwargs: None,
            "_salary_note": lambda *args, **kwargs: None,
            "_chart": lambda *args, **kwargs: None,
            "_SCATTER_MARKER": {},
            "CONTRACT_COLORS": CONTRACT_COLORS,
            "years_from_frame": years_from_frame,
            "teams_from_frame": teams_from_frame,
            "metric_label": metric_label,
            "scale_money_columns": scale_money_columns,
            "player_id_columns_for_duplicate_names": player_id_columns_for_duplicate_names,
            "nav_page": nav_page,
            "empty_state_copy": empty_state_copy,
            "salary_coverage_note": salary_coverage_note,
            "html": __import__("html"),
        },
    )

    namespace["page_player_explorer"]()

    assert any("Multiple players share a name" in cap for cap in st.captions)
    assert len(captured_tables) >= 4
    for table in captured_tables[:4]:
        assert "player_id" in table.columns


def test_team_deep_dive_roster_includes_player_id_for_name_collisions():
    st = _FakeStreamlit(selectbox_values={"tp_team": "Alpha"})
    metrics = pd.DataFrame(
        [
            {
                "team_name": "Alpha",
                "year_id": 2023,
                "wins": 82,
                "losses": 80,
                "run_diff": 10,
                "payroll": 100_000_000,
                "team_total_war": 35.0,
                "wins_per_10m": 8.2,
                "pythag_wins": 81.0,
            },
            {
                "team_name": "Alpha",
                "year_id": 2024,
                "wins": 90,
                "losses": 72,
                "run_diff": 55,
                "payroll": 120_000_000,
                "team_total_war": 42.0,
                "wins_per_10m": 7.5,
                "pythag_wins": 88.0,
            },
        ]
    )
    players = pd.DataFrame(
        [
            {
                "player_id": "dup-1",
                "name_full": "Chris Young",
                "year_id": 2024,
                "team_name": "Alpha",
                "player_type": "H",
                "pa": 250,
                "hr": 14,
                "bb": 20,
                "woba": 0.32,
                "batting_war": 1.5,
                "ip": 0.0,
                "era": 0.0,
                "fip": 0.0,
                "pitching_war": 0.0,
                "player_war": 1.5,
                "salary": 2_000_000,
                "surplus_value": 1_000_000,
                "contract_label": "fair_value",
            },
            {
                "player_id": "dup-2",
                "name_full": "Chris Young",
                "year_id": 2024,
                "team_name": "Alpha",
                "player_type": "P",
                "pa": 0,
                "hr": 0,
                "bb": 0,
                "woba": 0.0,
                "batting_war": 0.0,
                "ip": 70.0,
                "era": 3.4,
                "fip": 3.8,
                "pitching_war": 1.7,
                "player_war": 1.7,
                "salary": 3_000_000,
                "surplus_value": 1_100_000,
                "contract_label": "surplus_value",
            },
        ]
    )
    captured_tables: list[pd.DataFrame] = []

    def _fake_load(key: str):
        if key == "players":
            return players
        return None

    fake_px = SimpleNamespace(
        line=lambda *args, **kwargs: _FakeFigure(),
        bar=lambda *args, **kwargs: _FakeFigure(),
    )

    from dashboard.helpers import empty_state_copy, salary_coverage_note
    from src.baseball_analytics.dashboard_utils import player_id_columns_for_duplicate_names

    namespace = _load_app_symbols(
        functions=("page_team_deep_dive", "_page_header", "_empty", "_salary_note"),
        globals_dict={
            "pd": pd,
            "st": st,
            "metrics": metrics,
            "all_teams": ["Alpha"],
            "_season_picker": lambda *args, **kwargs: 2024,
            "_load": _fake_load,
            "_scale_payroll": lambda df: df,
            "_show_table": lambda df, *args, **kwargs: captured_tables.append(df.copy()),
            "_TEAM_COL_CFG": {},
            "_PLAYER_COL_CFG": {},
            "px": fake_px,
            "_chart": lambda *args, **kwargs: None,
            "_empty": lambda *args, **kwargs: None,
            "_page_header": lambda *args, **kwargs: None,
            "_salary_note": lambda *args, **kwargs: None,
            "scale_money_columns": scale_money_columns,
            "add_payroll_millions": add_payroll_millions,
            "format_money_millions": format_money_millions,
            "format_signed_int": format_signed_int,
            "format_war": format_war,
            "player_id_columns_for_duplicate_names": player_id_columns_for_duplicate_names,
            "nav_page": nav_page,
            "empty_state_copy": empty_state_copy,
            "salary_coverage_note": salary_coverage_note,
            "html": __import__("html"),
        },
    )

    namespace["page_team_deep_dive"]()

    roster_tables = [df for df in captured_tables if "name_full" in df.columns]
    assert roster_tables, "Expected at least one roster table render call"
    assert "player_id" in roster_tables[-1].columns
