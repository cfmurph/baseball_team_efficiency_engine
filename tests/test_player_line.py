"""Unit tests for documented player-line identities."""
from __future__ import annotations

import math

import pytest

from src.baseball_analytics.player_line import (
    apply_fielding_identities,
    apply_hitting_identities,
    apply_pitching_identities,
    apply_player_line_identities,
    as_number,
    babip,
    extra_base_hits,
    ground_fly,
    innings_per_start,
    isolated_power,
    outfield_assists,
    per_nine,
    range_factor,
    save_opportunities,
    singles,
    steal_pct,
    strikeout_walk_ratio,
    total_bases,
    total_chances,
    unearned_runs,
    win_pct,
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


def test_as_number_rejects_nan_and_garbage() -> None:
    assert as_number(None) is None
    assert as_number("") is None
    assert as_number("not-a-number") is None
    assert as_number(math.nan) is None
    assert as_number("12.5") == 12.5


def test_hitting_iso_babip_and_rates_need_every_input() -> None:
    assert isolated_power(0.71, 0.35) == pytest.approx(0.36)
    assert isolated_power(0.71, None) is None
    assert babip(140, 40, 400, 130, 5) == pytest.approx(100 / 235)
    assert babip(140, 40, 400, 130, None) is None
    assert babip(40, 40, 40, 0, 0) is None
    row = apply_hitting_identities(
        {
            "hits": 140,
            "doubles": 22,
            "triples": 1,
            "hr": 40,
            "ab": 400,
            "so": 130,
            "bb": 90,
            "pa": 500,
            "slg": 0.71,
            "avg": 0.35,
            "go": 90,
            "ao": 80,
        }
    )
    assert "babip" not in row
    assert row["iso"] == pytest.approx(0.36)
    assert row["k_pct"] == pytest.approx(130 / 500)
    assert row["bb_pct"] == pytest.approx(90 / 500)
    assert row["go_ao"] == pytest.approx(90 / 80)
    assert apply_hitting_identities({"so": 130, "pa": 0}).get("k_pct") is None


def test_pitching_identities_from_complete_counting() -> None:
    row = apply_pitching_identities(
        {
            "w": 12,
            "l": 8,
            "sv": 4,
            "bs": 1,
            "ip": 150,
            "gs": 27,
            "pitching_so": 145,
            "pitching_bb": 42,
            "pitching_hits": 120,
            "pitching_hr": 12,
            "pitching_r": 48,
            "er": 43,
            "pitching_go": 180,
            "pitching_ao": 120,
            "bf": 610,
        }
    )
    assert row["wpct"] == pytest.approx(0.6)
    assert row["svo"] == 5
    assert row["sv_pct"] == pytest.approx(0.8)
    assert row["uer"] == 5
    assert row["k9"] == pytest.approx(145 * 9 / 150)
    assert row["bb9"] == pytest.approx(42 * 9 / 150)
    assert row["h9"] == pytest.approx(120 * 9 / 150)
    assert row["hr9"] == pytest.approx(12 * 9 / 150)
    assert row["k_bb"] == pytest.approx(145 / 42)
    assert row["i_gs"] == pytest.approx(150 / 27)
    assert row["pitching_go_ao"] == pytest.approx(1.5)
    assert row["pitching_k_pct"] == pytest.approx(145 / 610)
    assert row["pitching_bb_pct"] == pytest.approx(42 / 610)


def test_pitching_rates_stay_empty_on_zero_or_missing_denominators() -> None:
    assert win_pct(0, 0) is None
    assert win_pct(12, None) is None
    assert per_nine(145, 0) is None
    assert per_nine(145, None) is None
    assert strikeout_walk_ratio(145, 0) is None
    assert innings_per_start(150, 0) is None
    assert ground_fly(180, 0) is None
    assert save_opportunities(4, None) is None
    row = apply_pitching_identities(
        {
            "w": 0,
            "l": 0,
            "sv": 2,
            "ip": 0,
            "gs": 0,
            "pitching_so": 3,
            "pitching_bb": 0,
            "pitching_hits": 1,
            "bf": 0,
            "pitching_ao": 0,
            "pitching_go": 4,
        }
    )
    assert "wpct" not in row
    assert "k9" not in row
    assert "k_bb" not in row
    assert "i_gs" not in row
    assert "pitching_go_ao" not in row
    assert "pitching_k_pct" not in row
    assert "sv_pct" not in row


def test_unearned_runs_floor_at_zero() -> None:
    assert unearned_runs(48, 43) == 5
    assert unearned_runs(40, 43) == 0
    assert unearned_runs(40, None) is None


def test_apply_pitching_does_not_overwrite_landed_values() -> None:
    row = apply_pitching_identities(
        {
            "w": 12,
            "l": 8,
            "sv": 4,
            "bs": 1,
            "ip": 150,
            "pitching_so": 145,
            "pitching_bb": 42,
            "wpct": 0.55,
            "k9": 9.9,
            "svo": 9,
        }
    )
    assert row["wpct"] == pytest.approx(0.55)
    assert row["k9"] == pytest.approx(9.9)
    assert row["svo"] == 9
    assert row["sv_pct"] == pytest.approx(4 / 9)


def test_ofa_only_for_outfield_and_cs_pct_ignores_batting_steals() -> None:
    assert outfield_assists(7, "RF") == 7
    assert outfield_assists(7, "lf") == 7
    assert outfield_assists(7, "SS") is None
    assert outfield_assists(7, "C") is None
    assert outfield_assists(None, "RF") is None

    batter = apply_fielding_identities(
        {
            "pa": 500,
            "hits": 140,
            "sb": 8,
            "cs": 3,
            "pos": "SS",
            "assists": 400,
            "putouts": 80,
            "errors": 12,
        }
    )
    assert "ofa" not in batter
    assert "cs_pct" not in batter
    assert batter["tc"] == 492

    catcher = apply_fielding_identities(
        {
            "fielding_pos": "C",
            "fielding_cs": 28,
            "fielding_sb": 42,
            "putouts": 900,
            "assists": 60,
            "errors": 4,
        }
    )
    assert catcher["cs_pct"] == pytest.approx(28 / 70)
    assert "ofa" not in catcher

    fielding_only = apply_fielding_identities({"cs": 10, "sb": 15})
    assert fielding_only["cs_pct"] == pytest.approx(10 / 25)
