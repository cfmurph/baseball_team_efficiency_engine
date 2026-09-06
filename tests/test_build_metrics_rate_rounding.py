"""Regression: Lahman rate columns must round when AB/PA are 0 or missing."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.transform.build_metrics import (
    _aggregate_lahman_batting,
    _aggregate_lahman_pitching,
    _round_numeric,
)

pytestmark = pytest.mark.unit


def test_round_numeric_accepts_pandas_na_without_raising() -> None:
    values = pd.Series([0.3334, pd.NA, 0.5], dtype="Float64")
    rounded = _round_numeric(values, 3)
    assert rounded.tolist()[0] == pytest.approx(0.333)
    assert pd.isna(rounded.tolist()[1])
    assert rounded.tolist()[2] == pytest.approx(0.5)


def test_aggregate_lahman_batting_rounds_rates_when_ab_or_pa_is_zero_or_missing() -> None:
    batting = pd.DataFrame(
        [
            {
                "playerID": "goodbat01",
                "yearID": 2024,
                "AB": 400,
                "H": 120,
                "2B": 20,
                "3B": 2,
                "HR": 15,
                "BB": 40,
                "HBP": 4,
                "SF": 4,
            },
            {
                "playerID": "zeroab01",
                "yearID": 2024,
                "AB": 0,
                "H": 0,
                "2B": 0,
                "3B": 0,
                "HR": 0,
                "BB": 0,
                "HBP": 0,
                "SF": 0,
            },
            {
                "playerID": "missab01",
                "yearID": 2024,
                "AB": pd.NA,
                "H": 1,
                "2B": 0,
                "3B": 0,
                "HR": 0,
                "BB": 0,
                "HBP": 0,
                "SF": 0,
            },
        ]
    )

    out = _aggregate_lahman_batting(batting)
    by_id = out.set_index("player_id")

    good = by_id.loc["goodbat01"]
    assert good["avg"] == pytest.approx(0.300)
    assert good["obp"] == pytest.approx(0.366)
    assert good["slg"] == pytest.approx(0.472)
    assert good["ops"] == pytest.approx(0.838)

    zero = by_id.loc["zeroab01"]
    assert pd.isna(zero["avg"])
    assert pd.isna(zero["obp"])
    assert pd.isna(zero["slg"])
    assert pd.isna(zero["ops"])

    missing = by_id.loc["missab01"]
    assert pd.isna(missing["avg"])
    assert pd.isna(missing["slg"])
    assert pd.isna(missing["ops"])


def test_aggregate_lahman_pitching_rounds_era_when_missing() -> None:
    pitching = pd.DataFrame(
        [
            {"playerID": "ace001", "yearID": 2024, "ERA": 2.456, "IP": 180, "ER": 49, "G": 30},
            {"playerID": "norate", "yearID": 2024, "ERA": pd.NA, "IP": 0, "ER": 0, "G": 1},
        ]
    )

    out = _aggregate_lahman_pitching(pitching)
    by_id = out.set_index("player_id")
    assert by_id.loc["ace001", "era"] == pytest.approx(2.46)
    assert pd.isna(by_id.loc["norate", "era"])
