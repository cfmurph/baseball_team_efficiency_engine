from __future__ import annotations

import pandas as pd

from src.baseball_analytics.metrics import pythagorean_wins, top_salary_shares, salary_concentration


def test_pythagorean_wins_length() -> None:
    wins = pythagorean_wins(pd.Series([700]), pd.Series([650]), pd.Series([162]))
    assert len(wins) == 1
    assert float(wins[0]) > 0


def test_top_salary_shares() -> None:
    df = pd.DataFrame({"salary": [10, 20, 30, 40, 50]})
    result = top_salary_shares(df)
    assert round(result["top_1_salary_share"], 4) == round(50 / 150, 4)
    assert round(result["top_5_salary_share"], 4) == 1.0


def test_salary_concentration() -> None:
    df = pd.DataFrame({"salary": [1, 1, 1, 10]})
    result = salary_concentration(df)
    assert result["gini_salary"] > 0
