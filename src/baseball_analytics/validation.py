"""
Lightweight data validation for pipeline artifacts.

Each check returns a ValidationResult. Run all checks via validate_all().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    rows_affected: int = 0


@dataclass
class ValidationReport:
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        level = logging.INFO if result.passed else logging.WARNING
        log.log(level, "[%s] %s — %s", "PASS" if result.passed else "FAIL", result.name, result.message)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def summary(self) -> str:
        total = len(self.results)
        failed = self.n_failed
        return f"{total - failed}/{total} checks passed" + (
            f" — {failed} FAILED" if failed else ""
        )


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------

def check_not_empty(df: pd.DataFrame, name: str) -> ValidationResult:
    ok = len(df) > 0
    return ValidationResult(name, ok, f"{len(df)} rows" if ok else "DataFrame is empty")


def check_no_nulls(df: pd.DataFrame, cols: list[str], name: str) -> ValidationResult:
    null_counts = df[cols].isnull().sum()
    bad = null_counts[null_counts > 0]
    ok = len(bad) == 0
    msg = "No nulls" if ok else f"Nulls in: {bad.to_dict()}"
    return ValidationResult(name, ok, msg, rows_affected=int(bad.sum()))


def check_column_range(
    df: pd.DataFrame,
    col: str,
    lo: float | None,
    hi: float | None,
    name: str,
) -> ValidationResult:
    if col not in df.columns:
        return ValidationResult(name, False, f"Column '{col}' missing from DataFrame")
    non_null = df[col].dropna()
    mask = pd.Series([True] * len(non_null), index=non_null.index)
    if lo is not None:
        mask &= non_null >= lo
    if hi is not None:
        mask &= non_null <= hi
    bad = (~mask).sum()
    ok = bad == 0
    msg = f"All {len(non_null)} non-null values in [{lo}, {hi}]" if ok else f"{bad} non-null values outside [{lo}, {hi}]"
    return ValidationResult(name, ok, msg, rows_affected=int(bad))


def check_no_duplicate_pk(df: pd.DataFrame, pk_cols: list[str], name: str) -> ValidationResult:
    dups = df.duplicated(subset=pk_cols).sum()
    ok = dups == 0
    return ValidationResult(name, ok, f"No duplicates" if ok else f"{dups} duplicate PKs on {pk_cols}", rows_affected=int(dups))


def check_referential_integrity(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    child_col: str,
    parent_col: str,
    name: str,
) -> ValidationResult:
    orphans = ~child_df[child_col].isin(parent_df[parent_col])
    bad = orphans.sum()
    ok = bad == 0
    return ValidationResult(name, ok, "All keys resolve" if ok else f"{bad} orphan keys", rows_affected=int(bad))


# ---------------------------------------------------------------------------
# Pipeline-specific validation suites
# ---------------------------------------------------------------------------

def validate_fact_team_season(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.add(check_not_empty(df, "fact_team_season not empty"))
    report.add(check_no_duplicate_pk(df, ["team_key", "season_key"], "fact_team_season PK unique"))
    # Allow shortened seasons (strikes: 1981, 1994-95, COVID 2020)
    report.add(check_column_range(df, "wins", 10, 130, "wins in plausible range"))
    report.add(check_column_range(df, "losses", 10, 125, "losses in plausible range"))
    report.add(check_column_range(df, "payroll", 0, None, "payroll non-negative"))
    report.add(check_column_range(df, "pythag_wins", 10, 150, "pythag_wins plausible"))
    if "gini_salary" in df.columns:
        report.add(check_column_range(df, "gini_salary", 0, 1, "gini_salary in [0,1]"))
    return report


def validate_fact_player_season(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.add(check_not_empty(df, "fact_player_season not empty"))
    report.add(check_no_duplicate_pk(df, ["player_id", "season_key", "team_id"], "fact_player_season PK unique"))
    report.add(check_column_range(df, "salary", 0, None, "salary non-negative"))
    return report


def validate_dim_team(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()
    report.add(check_not_empty(df, "dim_team not empty"))
    report.add(check_no_duplicate_pk(df, ["team_key"], "dim_team PK unique"))
    return report


def validate_all(
    fact_team: pd.DataFrame,
    fact_player: pd.DataFrame,
    dim_team: pd.DataFrame,
) -> ValidationReport:
    combined = ValidationReport()
    for sub_report in [
        validate_fact_team_season(fact_team),
        validate_fact_player_season(fact_player),
        validate_dim_team(dim_team),
    ]:
        for result in sub_report.results:
            combined.add(result)
    log.info("Validation complete: %s", combined.summary())
    return combined
