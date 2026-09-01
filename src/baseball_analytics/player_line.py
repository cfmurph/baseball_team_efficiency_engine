"""Documented player-line identities from landed counting.

Used by the published player projection so GET /v1/players never invents
plus-stats, Statcast, or league-average fakes. A derived column is filled
only when every required input is present; missing inputs stay missing.
"""
from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

_OF_POS = frozenset({"OF", "LF", "CF", "RF"})


def as_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def json_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return float(value)


def _all_present(*values: object) -> bool:
    return all(as_number(value) is not None for value in values)


def singles(hits: object, doubles: object, triples: object, hr: object) -> float | None:
    """1B = H − 2B − 3B − HR when those four exist."""
    if not _all_present(hits, doubles, triples, hr):
        return None
    return max(0.0, as_number(hits) - as_number(doubles) - as_number(triples) - as_number(hr))


def extra_base_hits(doubles: object, triples: object, hr: object) -> float | None:
    """XBH = 2B + 3B + HR."""
    if not _all_present(doubles, triples, hr):
        return None
    return as_number(doubles) + as_number(triples) + as_number(hr)


def total_bases(hits: object, doubles: object, triples: object, hr: object) -> float | None:
    """TB = H + 2B + 2×3B + 3×HR (= 1B + 2×2B + 3×3B + 4×HR)."""
    if not _all_present(hits, doubles, triples, hr):
        return None
    return (
        as_number(hits)
        + as_number(doubles)
        + 2.0 * as_number(triples)
        + 3.0 * as_number(hr)
    )


def steal_pct(sb: object, cs: object) -> float | None:
    """SB% = SB / (SB + CS). Needs CS."""
    if not _all_present(sb, cs):
        return None
    denom = as_number(sb) + as_number(cs)
    if denom <= 0:
        return None
    return as_number(sb) / denom


def total_chances(putouts: object, assists: object, errors: object) -> float | None:
    """TC = PO + A + E."""
    if not _all_present(putouts, assists, errors):
        return None
    return as_number(putouts) + as_number(assists) + as_number(errors)


def range_factor(putouts: object, assists: object, inn: object) -> float | None:
    """RF = 9 × (PO + A) / INN."""
    if not _all_present(putouts, assists, inn):
        return None
    innings = as_number(inn)
    if innings is None or innings <= 0:
        return None
    return 9.0 * (as_number(putouts) + as_number(assists)) / innings


def isolated_power(slg: object, avg: object) -> float | None:
    """ISO = SLG − AVG."""
    if not _all_present(slg, avg):
        return None
    return as_number(slg) - as_number(avg)


def babip(hits: object, hr: object, ab: object, so: object, sf: object) -> float | None:
    """BABIP = (H − HR) / (AB − SO − HR + SF). Needs SF."""
    if not _all_present(hits, hr, ab, so, sf):
        return None
    denom = as_number(ab) - as_number(so) - as_number(hr) + as_number(sf)
    if denom <= 0:
        return None
    return (as_number(hits) - as_number(hr)) / denom


def per_nine(count: object, ip: object) -> float | None:
    if not _all_present(count, ip):
        return None
    innings = as_number(ip)
    if innings is None or innings <= 0:
        return None
    return as_number(count) * 9.0 / innings


def strikeout_walk_ratio(so: object, bb: object) -> float | None:
    if not _all_present(so, bb):
        return None
    walks = as_number(bb)
    if walks is None or walks <= 0:
        return None
    return as_number(so) / walks


def rate_pct(count: object, opportunities: object) -> float | None:
    if not _all_present(count, opportunities):
        return None
    denom = as_number(opportunities)
    if denom is None or denom <= 0:
        return None
    return as_number(count) / denom


def win_pct(wins: object, losses: object) -> float | None:
    """WPCT = W / (W + L)."""
    if not _all_present(wins, losses):
        return None
    denom = as_number(wins) + as_number(losses)
    if denom <= 0:
        return None
    return as_number(wins) / denom


def innings_per_start(ip: object, gs: object) -> float | None:
    """I/GS = IP / GS."""
    if not _all_present(ip, gs):
        return None
    starts = as_number(gs)
    if starts is None or starts <= 0:
        return None
    return as_number(ip) / starts


def ground_fly(go: object, ao: object) -> float | None:
    """GO/AO only when both groundout and flyout counts exist."""
    if not _all_present(go, ao):
        return None
    fly = as_number(ao)
    if fly is None or fly <= 0:
        return None
    return as_number(go) / fly


def save_opportunities(saves: object, blown: object) -> float | None:
    """SVO = SV + BS when both counts exist."""
    if not _all_present(saves, blown):
        return None
    return as_number(saves) + as_number(blown)


def save_pct(saves: object, svo: object) -> float | None:
    return rate_pct(saves, svo)


def unearned_runs(runs: object, er: object) -> float | None:
    if not _all_present(runs, er):
        return None
    return max(0.0, as_number(runs) - as_number(er))


def catcher_cs_pct(caught: object, stolen: object) -> float | None:
    """CS% = CS / (CS + SB allowed)."""
    if not _all_present(caught, stolen):
        return None
    denom = as_number(caught) + as_number(stolen)
    if denom <= 0:
        return None
    return as_number(caught) / denom


def outfield_assists(assists: object, pos: object) -> float | None:
    """OFA is assists for OF / LF / CF / RF. Infield assists stay off this column."""
    if as_number(assists) is None:
        return None
    key = str(pos or "").strip().upper()
    if key not in _OF_POS:
        return None
    return as_number(assists)


def _missing(row: Mapping[str, Any], key: str) -> bool:
    return row.get(key) in (None, "")


def _set_if_missing(row: MutableMapping[str, Any], key: str, value: float | None) -> None:
    if not _missing(row, key):
        return
    if value is None:
        return
    row[key] = json_number(value)


def apply_hitting_identities(row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    hits = row.get("hits", row.get("h"))
    doubles = row.get("doubles")
    triples = row.get("triples")
    hr = row.get("hr")
    _set_if_missing(row, "singles", singles(hits, doubles, triples, hr))
    _set_if_missing(row, "xbh", extra_base_hits(doubles, triples, hr))
    _set_if_missing(row, "tb", total_bases(hits, doubles, triples, hr))
    _set_if_missing(row, "sb_pct", steal_pct(row.get("sb"), row.get("cs")))
    _set_if_missing(row, "go_ao", ground_fly(row.get("go"), row.get("ao")))
    _set_if_missing(row, "iso", isolated_power(row.get("slg"), row.get("avg")))
    _set_if_missing(
        row,
        "babip",
        babip(hits, hr, row.get("ab"), row.get("so"), row.get("sf")),
    )
    pa = row.get("pa")
    _set_if_missing(row, "k_pct", rate_pct(row.get("so"), pa))
    _set_if_missing(row, "bb_pct", rate_pct(row.get("bb"), pa))
    return row


def apply_pitching_identities(row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    ip = row.get("ip")
    so = row.get("pitching_so", row.get("so"))
    bb = row.get("pitching_bb", row.get("bb"))
    hits = row.get("pitching_hits")
    hr = row.get("pitching_hr")
    _set_if_missing(row, "wpct", win_pct(row.get("w"), row.get("l")))
    if _missing(row, "svo"):
        _set_if_missing(row, "svo", save_opportunities(row.get("sv"), row.get("bs")))
    _set_if_missing(row, "sv_pct", save_pct(row.get("sv"), row.get("svo")))
    _set_if_missing(row, "uer", unearned_runs(row.get("pitching_r"), row.get("er")))
    _set_if_missing(row, "pitching_go_ao", ground_fly(row.get("pitching_go"), row.get("pitching_ao")))
    _set_if_missing(row, "k9", per_nine(so, ip))
    _set_if_missing(row, "bb9", per_nine(bb, ip))
    _set_if_missing(row, "h9", per_nine(hits, ip))
    _set_if_missing(row, "hr9", per_nine(hr, ip))
    _set_if_missing(row, "k_bb", strikeout_walk_ratio(so, bb))
    bf = row.get("bf")
    _set_if_missing(row, "pitching_k_pct", rate_pct(so, bf))
    _set_if_missing(row, "pitching_bb_pct", rate_pct(bb, bf))
    _set_if_missing(row, "i_gs", innings_per_start(ip, row.get("gs")))
    return row


def apply_fielding_identities(row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    po = row.get("po", row.get("putouts"))
    assists = row.get("a", row.get("assists"))
    errors = row.get("e", row.get("errors"))
    inn = row.get("inn", row.get("fielding_inn"))
    _set_if_missing(row, "tc", total_chances(po, assists, errors))
    _set_if_missing(row, "rf", range_factor(po, assists, inn))
    pos = row.get("pos") or row.get("fielding_pos")
    if _missing(row, "ofa"):
        ofa = outfield_assists(assists, pos)
        if ofa is not None:
            row["ofa"] = json_number(ofa)
    caught = row.get("fielding_cs")
    stolen = row.get("fielding_sb")
    if caught is None and stolen is None and row.get("pa") is None and row.get("hits") is None:
        caught = row.get("cs")
        stolen = row.get("sb")
    _set_if_missing(row, "cs_pct", catcher_cs_pct(caught, stolen))
    return row


def apply_player_line_identities(row: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    apply_hitting_identities(row)
    apply_pitching_identities(row)
    apply_fielding_identities(row)
    return row
