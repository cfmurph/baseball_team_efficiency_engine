"""Unit tests for documented player-line identities."""
from __future__ import annotations

import pytest

from src.baseball_analytics.player_line import (
    apply_hitting_identities,
    apply_player_line_identities,
    extra_base_hits,
    range_factor,
    singles,
    steal_pct,
    total_bases,
    total_chances,
)

pytestmark = pytest.mark.unit


def test_singles_xbh_tb_from_complete_counting() -> None:
    assert singles(140, 22, 1, 40) == 77
    assert extra_base_hits(22, 1, 40) == 63
    assert total_bases(140, 22, 1, 40) == 284


def test_identities_stay_empty_when_an_input_is_missing() -> None:
    assert singles(140, 22, None, 40) is None
    assert extra_base_hits(22, None, 40) is None
    assert total_bases(140, None, 1, 40) is None
    assert steal_pct(8, None) is None
    assert total_chances(248, 7, None) is None
    assert range_factor(248, 7, None) is None
    row = apply_hitting_identities({"hits": 140, "doubles": 22, "hr": 40})
    assert "singles" not in row
    assert "xbh" not in row
    assert "tb" not in row


def test_sb_pct_tc_rf() -> None:
    assert steal_pct(8, 3) == pytest.approx(8 / 11)
    assert total_chances(248, 7, 3) == 258
    assert range_factor(248, 7, 980) == pytest.approx(9 * 255 / 980)


def test_apply_does_not_overwrite_landed_values() -> None:
    row = apply_player_line_identities(
        {
            "hits": 140,
            "doubles": 22,
            "triples": 1,
            "hr": 40,
            "singles": 80,
            "putouts": 248,
            "assists": 7,
            "errors": 3,
            "fielding_inn": 980,
            "fielding_pos": "RF",
        }
    )
    assert row["singles"] == 80
    assert row["xbh"] == 63
    assert row["tc"] == 258
    assert row["ofa"] == 7
