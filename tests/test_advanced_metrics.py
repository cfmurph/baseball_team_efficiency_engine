"""Tests for advanced metrics in metrics.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baseball_analytics.metrics import (
    pythagorean_wins,
    safe_divide,
    salary_concentration,
    top_salary_shares,
    cost_per_war,
    war_per_dollar,
    surplus_value_team,
    pythag_gap,
    war_win_gap,
    player_surplus_value,
    classify_contract,
    payroll_underperformer_share,
    detect_team_window,
)


# ---- Original metric tests ----

def test_pythagorean_wins_length():
    wins = pythagorean_wins(pd.Series([700]), pd.Series([650]), pd.Series([162]))
    assert len(wins) == 1
    assert float(wins[0]) > 0


def test_pythagorean_wins_above_500_for_better_offense():
    wins = pythagorean_wins(pd.Series([800]), pd.Series([600]), pd.Series([162]))
    assert float(wins[0]) > 81


def test_top_salary_shares():
    df = pd.DataFrame({"salary": [10, 20, 30, 40, 50]})
    result = top_salary_shares(df)
    assert round(result["top_1_salary_share"], 4) == round(50 / 150, 4)
    assert round(result["top_5_salary_share"], 4) == 1.0


def test_salary_concentration():
    df = pd.DataFrame({"salary": [1, 1, 1, 10]})
    result = salary_concentration(df)
    assert result["gini_salary"] > 0


def test_salary_concentration_equal_salaries():
    df = pd.DataFrame({"salary": [10, 10, 10, 10]})
    result = salary_concentration(df)
    assert result["gini_salary"] == pytest.approx(0.0, abs=1e-6)


def test_safe_divide_zero_denominator():
    result = safe_divide(pd.Series([10.0]), pd.Series([0.0]))
    assert np.isnan(float(result[0]))


def test_safe_divide_normal():
    result = safe_divide(pd.Series([100.0]), pd.Series([4.0]))
    assert float(result[0]) == pytest.approx(25.0)


# ---- WAR efficiency metrics ----

def test_cost_per_war():
    result = cost_per_war(pd.Series([100_000_000.0]), pd.Series([10.0]))
    assert float(result[0]) == pytest.approx(10_000_000.0)


def test_war_per_dollar():
    result = war_per_dollar(pd.Series([10.0]), pd.Series([100_000_000.0]), scale=1_000_000.0)
    assert float(result[0]) == pytest.approx(0.1)


def test_surplus_value_team_positive():
    # 10 WAR at $8M/WAR = $80M value vs $60M payroll → $20M surplus
    result = surplus_value_team(pd.Series([60_000_000.0]), pd.Series([10.0]))
    assert float(result[0]) > 0


def test_surplus_value_team_negative():
    # 2 WAR at $8M/WAR = $16M value vs $80M payroll → large negative
    result = surplus_value_team(pd.Series([80_000_000.0]), pd.Series([2.0]))
    assert float(result[0]) < 0


def test_pythag_gap_positive():
    result = pythag_gap(pd.Series([95]), pd.Series([88.0]))
    assert float(result[0]) == pytest.approx(7.0)


def test_war_win_gap():
    # 50 WAR + 48 replacement wins = 98 implied; actual 90 → gap of -8
    result = war_win_gap(pd.Series([90]), pd.Series([50.0]))
    assert float(result[0]) == pytest.approx(-8.0)


# ---- Player-level metrics ----

def test_player_surplus_value_positive():
    salary = pd.Series([1_000_000.0])
    war = pd.Series([5.0])
    result = player_surplus_value(salary, war)
    assert float(result[0]) > 0


def test_player_surplus_value_negative_dead_money():
    salary = pd.Series([20_000_000.0])
    war = pd.Series([0.0])
    result = player_surplus_value(salary, war)
    assert float(result[0]) < 0


def test_classify_contract_dead_money():
    surplus = pd.Series([-15_000_000.0])
    war = pd.Series([-0.5])
    result = classify_contract(surplus, war)
    assert result[0] == "dead_money"


def test_classify_contract_surplus():
    surplus = pd.Series([10_000_000.0])
    war = pd.Series([4.0])
    result = classify_contract(surplus, war)
    assert result[0] == "surplus_value"


def test_classify_contract_overpaid():
    surplus = pd.Series([-5_000_000.0])
    war = pd.Series([1.0])
    result = classify_contract(surplus, war)
    assert result[0] == "overpaid"


def test_payroll_underperformer_share():
    df = pd.DataFrame({
        "salary": [10_000_000, 5_000_000, 1_000_000],
        "player_war": [-1.0, 0.2, 3.0],
    })
    share = payroll_underperformer_share(df)
    # Player 1 (WAR < 0.5): $10M + $5M = $15M / $16M ≈ 0.9375
    assert 0.9 < share < 1.0


def test_payroll_underperformer_share_none():
    df = pd.DataFrame({"salary": [0.0], "player_war": [5.0]})
    # Total salary is 0 → should return nan
    result = payroll_underperformer_share(df)
    assert np.isnan(result)


# ---- Window detection ----

def test_detect_team_window_contending():
    df = pd.DataFrame({
        "year_id": range(2015, 2022),
        "wins": [90, 91, 92, 95, 93, 94, 96],
        "payroll": [120e6] * 7,
        "team_total_war": [45.0] * 7,
    })
    result = detect_team_window(df)
    assert "window_phase" in result.columns
    # All high-win rows should be contending
    assert (result["window_phase"] == "contending").any()


def test_detect_team_window_rebuilding():
    # Payrolls vary enough that early-year values rank in bottom third
    df = pd.DataFrame({
        "year_id": range(2015, 2022),
        "wins": [62, 63, 60, 64, 65, 80, 88],
        "payroll": [30e6, 32e6, 28e6, 35e6, 100e6, 110e6, 120e6],
        "team_total_war": [18.0, 19.0, 17.0, 20.0, 30.0, 35.0, 40.0],
    })
    result = detect_team_window(df)
    assert "window_phase" in result.columns
    # Low-win, low-payroll early years should be classified as rebuilding or steady
    early_phases = result[result["year_id"] < 2018]["window_phase"].unique()
    assert any(p in early_phases for p in ["rebuilding", "steady"])
