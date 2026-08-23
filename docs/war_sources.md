# WAR sources

## Real WAR (preferred)

**Source:** Baseball-Reference rWAR (also called bWAR / rWAR).

Official daily files, no API key:

| Role | URL |
|---|---|
| Batting | https://www.baseball-reference.com/data/war_daily_bat.txt |
| Pitching | https://www.baseball-reference.com/data/war_daily_pitch.txt |

**Coverage:** 1871–present (files are updated daily). This project filters to `min_year` (1990 by default).

**Grain:** one row per player × year × team stint. Stints on the same team are summed. There are no combined `TOT` rows.

FanGraphs fWAR was considered. BR is the cleaner fit here: published CSVs, player–team–stint grain, and Lahman already ships a `bbrefID` crosswalk. fWAR can still be added later the same way (new extract + Chadwick `key_fangraphs`).

## Player ID mapping

Lahman `playerID` is **not** always equal to the Baseball-Reference ID (`~500` mismatches in current People.csv, plus nulls).

1. **Primary:** `People.bbrefID` → BR `player_ID` → Lahman `playerID`
2. **Fallback:** use BR `player_ID` as `playerID` when `bbrefID` is missing

Chadwick register is not required for rWAR. It is the right tool if we later ingest FanGraphs IDs (`key_fangraphs`).

## Team ID mapping

BR uses modern abbreviations (`NYY`, `LAD`, `NYM`). Lahman uses older codes (`NYA`, `LAN`, `NYN`). A few franchises also change IDs by year (Brewers `MIL`→`ML4` before 1998; Rays `TBD`/`TBR`→`TBA`; Nationals `WSN`→`WAS`).

Crosswalk: `data/crosswalks/br_team_map.csv` (year-aware).

If a player-year is unique on both sides but the team ID still misses, WAR is attached by `playerID + year` only. Traded players are never allocated this way.

## Approximate WAR (fallback)

Lahman-only wOBA batting WAR and FIP pitching WAR in `src/baseball_analytics/war.py`. Used when no rWAR row joins.

`war_source` on `fact_player_season` is `real` or `approx`.
`war_source` on `fact_team_season` is `real`, `approx`, or `mixed`.

Cost-per-WAR, surplus value, dead money, and `team_total_war` all use the **effective** WAR (real when present).
