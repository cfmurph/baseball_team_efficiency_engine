"""Tests for the data validation module."""
from __future__ import annotations

import pandas as pd
import pytest

from src.baseball_analytics.validation import (
    ValidationResult,
    ValidationReport,
    check_not_empty,
    check_no_nulls,
    check_column_range,
    check_no_duplicate_pk,
    check_referential_integrity,
    validate_fact_team_season,
    validate_dim_team,
)


# ---- Unit checks ----

def test_check_not_empty_passes():
    df = pd.DataFrame({"a": [1, 2]})
    result = check_not_empty(df, "test")
    assert result.passed


def test_check_not_empty_fails():
    df = pd.DataFrame()
    result = check_not_empty(df, "test")
    assert not result.passed


def test_check_no_nulls_passes():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = check_no_nulls(df, ["a", "b"], "test")
    assert result.passed


def test_check_no_nulls_fails():
    df = pd.DataFrame({"a": [1, None]})
    result = check_no_nulls(df, ["a"], "test")
    assert not result.passed
    assert result.rows_affected == 1


def test_check_column_range_passes():
    df = pd.DataFrame({"wins": [80, 90, 100]})
    result = check_column_range(df, "wins", 40, 130, "wins range")
    assert result.passed


def test_check_column_range_fails():
    df = pd.DataFrame({"wins": [80, 200]})
    result = check_column_range(df, "wins", 40, 130, "wins range")
    assert not result.passed
    assert result.rows_affected == 1


def test_check_column_range_missing_col():
    df = pd.DataFrame({"other": [1]})
    result = check_column_range(df, "wins", 0, 162, "wins range")
    assert not result.passed


def test_check_no_duplicate_pk_passes():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = check_no_duplicate_pk(df, ["a", "b"], "pk")
    assert result.passed


def test_check_no_duplicate_pk_fails():
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "x"]})
    result = check_no_duplicate_pk(df, ["a", "b"], "pk")
    assert not result.passed
    assert result.rows_affected == 1


def test_check_referential_integrity_passes():
    child = pd.DataFrame({"team_id": ["NYA", "BOS"]})
    parent = pd.DataFrame({"team_id": ["NYA", "BOS", "LAA"]})
    result = check_referential_integrity(child, parent, "team_id", "team_id", "fk")
    assert result.passed


def test_check_referential_integrity_fails():
    child = pd.DataFrame({"team_id": ["NYA", "UNKNOWN"]})
    parent = pd.DataFrame({"team_id": ["NYA", "BOS"]})
    result = check_referential_integrity(child, parent, "team_id", "team_id", "fk")
    assert not result.passed
    assert result.rows_affected == 1


# ---- Report aggregation ----

def test_validation_report_all_pass():
    report = ValidationReport()
    report.add(ValidationResult("a", True, "ok"))
    report.add(ValidationResult("b", True, "ok"))
    assert report.passed
    assert report.n_failed == 0


def test_validation_report_partial_fail():
    report = ValidationReport()
    report.add(ValidationResult("a", True, "ok"))
    report.add(ValidationResult("b", False, "fail"))
    assert not report.passed
    assert report.n_failed == 1


def test_validation_report_summary_format():
    report = ValidationReport()
    report.add(ValidationResult("a", True, "ok"))
    report.add(ValidationResult("b", False, "fail"))
    summary = report.summary()
    assert "1/2" in summary
    assert "FAILED" in summary


# ---- Suite tests ----

def test_validate_fact_team_season_passes():
    df = pd.DataFrame({
        "team_key": ["NYA_2010", "BOS_2010"],
        "season_key": [2010, 2010],
        "wins": [95, 89],
        "losses": [67, 73],
        "payroll": [200e6, 160e6],
        "pythag_wins": [93.0, 87.5],
        "gini_salary": [0.4, 0.35],
    })
    report = validate_fact_team_season(df)
    assert report.passed, report.summary()


def test_validate_fact_team_season_bad_wins():
    df = pd.DataFrame({
        "team_key": ["NYA_2010"],
        "season_key": [2010],
        "wins": [200],   # clearly wrong (> 130)
        "losses": [67],
        "payroll": [200e6],
        "pythag_wins": [93.0],
    })
    report = validate_fact_team_season(df)
    assert not report.passed


def test_validate_dim_team_passes():
    df = pd.DataFrame({
        "team_key": ["NYA", "BOS"],
        "team_id": ["NYA", "BOS"],
        "franchise_id": ["NYY", "BRS"],
        "team_name": ["New York Yankees", "Boston Red Sox"],
        "league_id": ["AL", "AL"],
    })
    report = validate_dim_team(df)
    assert report.passed
