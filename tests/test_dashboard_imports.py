"""Guard that dashboard local imports work without a pre-set PYTHONPATH."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
ROOT = APP_PATH.resolve().parents[1]


def _is_local_import(node: ast.AST) -> str | None:
    if isinstance(node, ast.ImportFrom) and node.module:
        module = node.module
        if module == "src" or module.startswith("src.") or module == "dashboard" or module.startswith("dashboard."):
            return module
    if isinstance(node, ast.Import):
        for alias in node.names:
            name = alias.name
            if name == "src" or name.startswith("src.") or name == "dashboard" or name.startswith("dashboard."):
                return name
    return None


def _is_path_bootstrap(node: ast.AST) -> bool:
    return isinstance(node, ast.If) and "sys.path.insert" in ast.unparse(node)


def test_sys_path_bootstrap_precedes_local_imports() -> None:
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    saw_bootstrap = False
    for node in tree.body:
        if _is_path_bootstrap(node):
            saw_bootstrap = True
            continue
        module = _is_local_import(node)
        if module is not None:
            assert saw_bootstrap, f"{module} is imported before the sys.path bootstrap"
    assert saw_bootstrap, "dashboard/app.py must insert the repo root onto sys.path"


def test_dashboard_modules_import_without_pythonpath() -> None:
    """Reproduce Streamlit's sys.path (script dir first, no repo root)."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    code = r"""
import sys
from pathlib import Path

script_dir = Path("dashboard").resolve()
root = script_dir.parent
sys.path = [str(script_dir)] + [
    p for p in sys.path if Path(p).resolve() != root
]

_ROOT = Path("dashboard/app.py").resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.baseball_analytics.dashboard_helpers import (
    apply_layout_and_render_chart,
    compute_slider_max,
)
from src.baseball_analytics.dashboard_utils import (
    apply_plotly_layout,
    calculate_slider_max,
    player_id_columns_for_duplicate_names,
    render_plotly_chart,
    scale_payroll_for_display,
)
from dashboard.helpers import format_war

assert callable(compute_slider_max)
assert callable(apply_layout_and_render_chart)
assert callable(apply_plotly_layout)
assert callable(calculate_slider_max)
assert callable(player_id_columns_for_duplicate_names)
assert callable(render_plotly_chart)
assert callable(scale_payroll_for_display)
assert callable(format_war)
print("ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
