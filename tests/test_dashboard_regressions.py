"""Regression tests for recently fixed dashboard guard paths."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import plotly.graph_objects as go


DASHBOARD_APP_PATH = Path("dashboard/app.py")


def _read_dashboard_ast() -> ast.Module:
    source = DASHBOARD_APP_PATH.read_text(encoding="utf-8")
    return ast.parse(source, filename=str(DASHBOARD_APP_PATH))


def _extract_chart_namespace() -> dict:
    """Load only chart-related constants/functions without importing full app."""
    module = _read_dashboard_ast()
    needed_nodes: list[ast.AST] = []
    for node in module.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_PLOTLY_LAYOUT" in targets:
                needed_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in {"_apply_layout", "_chart"}:
            needed_nodes.append(node)

    mini_module = ast.Module(body=needed_nodes, type_ignores=[])
    ast.fix_missing_locations(mini_module)
    ns: dict = {}
    exec(compile(mini_module, str(DASHBOARD_APP_PATH), "exec"), ns)
    return ns


def _get_slider_max_expr() -> ast.AST:
    module = _read_dashboard_ast()
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "_slider_max" in targets:
            return node.value
    raise AssertionError("Could not find _slider_max assignment in dashboard/app.py")


def test_chart_renders_once_with_expected_layout_and_height():
    """_chart should render through Streamlit, not recurse into itself."""
    ns = _extract_chart_namespace()
    plotly_chart = MagicMock()
    ns["st"] = SimpleNamespace(plotly_chart=plotly_chart)

    fig = go.Figure()
    ns["_chart"](fig, height=333)

    assert fig.layout.height == 333
    assert fig.layout.plot_bgcolor == "#0d1117"
    plotly_chart.assert_called_once_with(fig, use_container_width=True)


def test_slider_max_expression_is_safe_for_empty_years():
    """Regression guard: empty all_years must not index all_years[-1]."""
    expr = _get_slider_max_expr()
    compiled = compile(ast.Expression(expr), str(DASHBOARD_APP_PATH), "eval")

    # Empty data should safely fall back to current year.
    assert eval(compiled, {"all_years": [], "_current_year": 2026}) == 2026
    # Non-empty data should still choose the max between latest data and current year.
    assert eval(compiled, {"all_years": [2016, 2024], "_current_year": 2023}) == 2024
