# ADR 0002 — Source-of-truth map

- Status: Accepted
- Date: 2026-08-23
- Related: `docs/war_sources.md`, #108, #105

## Decision

| Domain | Source of truth | Notes |
|---|---|---|
| WAR | Baseball-Reference rWAR (`pull_war`) | Warehouse `war_source=real` overlays rWAR. Fantasy cards publish the same value as `war_source=bbref`. |
| Majors team / player / game / season stats | MLB Stats API | Landed by #108 into `{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/`. |
| Live player-game spine | SportsDataIO | Phase 0 schema v0.1 (#128). Raw: `{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/`. |
| Player / team IDs and salary | Lahman | `People.bbrefID` joins BR. Salary coverage ends ~2016. |

## No dual-write WAR

`player_war` / `war` on exported metrics is the **effective** WAR: rWAR when
the BR join hits, otherwise the Lahman wOBA/FIP approximation. Do not store a
second competing WAR series, and do not replace BR rWAR with Stats API or
Sportradar WAR.

Warehouse grain keeps `war_source` as `real` | `approx` | `mixed`.
Card JSONL maps `real` → `edge.war_source=bbref` so the lake enum stays
`bbref` | `approx`. Never emit `fangraphs` until FG WAR ingest exists.

## Consequences

- `#108` may add Stats API facts for team/player/game/season; it must not
  overwrite `fact_player_season.player_war` from BR.
- Lahman remains the ID/salary bridge until a later ADR replaces it.
- Sportradar stays optional enrichment, never the WAR SoT.
