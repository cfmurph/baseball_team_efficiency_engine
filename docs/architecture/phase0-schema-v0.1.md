# Phase 0 schema v0.1 — SportsDataIO ingest

- Status: **LOCKED by Cole 2026-08-23**
- Issue: [#128](https://github.com/cfmurph/baseball_team_efficiency_engine/issues/128)
- Machine-readable twin: [phase0-schema-v0.1.json](phase0-schema-v0.1.json)

Do not widen this contract in a follow-up PR without a new schema version.
QA is sequenced after the first ingest PR.

## Decision

SportsDataIO is the **primary live ingest**. Internal primary keys are
stable UUIDs (UUID5 from SDIO ids). Live join keys are the SportsDataIO
integers (`sdio_player_id`, `sdio_team_id`, `sdio_game_id`) stored on
`external_id_alias` with `system = sportsdataio`.

Lahman, Baseball-Reference rWAR, and the MLB Stats API path from #123 stay
in place. This spine sits beside them; it does not replace them.

## Lake

Raw landing is a sibling of `runs/{run_id}/` and `current/`, not a second
layout:

```text
{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/…json
data/raw/sportsdataio/{endpoint}/{as_of_date}/…json
```

`ARTIFACTS_URI` stays vendor-agnostic (`file://` is the CI / QA path).
Re-pull of the same `as_of_date` overwrites that date partition only
(idempotent). `latest/` is not used for raw.

Published metrics and BenchOrStart cards stay on the #119 contract:

```text
runs/{run_id}/ + current/
current/fantasy/cards.jsonl
```

## Alias systems

`external_id_alias.system` is one of:

| system | meaning |
|---|---|
| `sportsdataio` | SDIO `PlayerID` / `TeamID` / `GameID` (primary live key) |
| `mlb` | MLB AM / Stats API id when SDIO exposes `MLBAMID` / `MlbID` |
| `bbref` | Baseball-Reference id when Lahman `People.bbrefID` joins via MLB id |
| `fangraphs` | reserved; attach only when an FG id is present |
| `lahman` | Lahman `playerID` / `teamID` when a People / team-map join hits |

Players and teams are **bootstrapped by SDIO id**. MLB and bbref aliases
are attached when available. Missing aliases are not an error.

## Spine grain

| Table | PK | Role |
|---|---|---|
| `player` | `player_id` (UUID) | Internal player identity |
| `team` | `team_id` (UUID) | Internal team identity |
| `game` | `game_id` (UUID) | Internal game identity |
| `external_id_alias` | `alias_id` (UUID); unique `(system, entity_type, external_id)` | Crosswalk |
| `player_game_stat` | `(player_id, game_id)` | Live fact spine |
| `player_season_stat` | `(player_id, season, team_id)` | Thin season rollup stub |

`entity_type` is `player` | `team` | `game`.

There are **no** `fantasy_*_stat` or `scout_*_stat` forked tables.
Fantasy, scout, and operator surfaces share this spine.

## Provenance (required on facts)

Every spine fact carries:

- `source` — `sportsdataio` for this ingest
- `source_endpoint` — lake endpoint token (`teams`, `players`, `games_by_date`, `player_game_stats`, …)
- `computed_at` — UTC timestamp when the row was built
- `as_of` — lake `as_of_date` (`YYYY-MM-DD`)
- `run_id` — nightly / extract run id when known
- `is_approx` — `false` for landed SDIO counting stats

Identity and alias rows carry the same columns so a later rebuild can
explain where a UUID came from.

## Account type (schema-only)

Reserved enum for later product surfaces:

- `fantasy`
- `scout`
- `operator_api`

Full RBAC is deferred. v0.1 does not enforce grants.

## Soft-fail

`SPORTSDATAIO_API_KEY` is read from the environment only (never hardcoded).
When the key is missing or an endpoint blips, the extract writes
`extract_report.json` with `soft_fail: true` and exits 0 so CI / nightly
without the secret still pass. The warehouse skips the spine when raw is
empty (same pattern as MLB Stats API #123).

## Non-goals

- Forked `fantasy_*_stat` / `scout_*_stat` tables
- Removing Lahman, BR rWAR, or MLB Stats API ingest
- Changing `current/fantasy/cards.jsonl` or `share.stat_line`
  (`+X.X edge · NN% conf` for bbref/real; approx hides conf; never `vs repl`)
- Thin API (#106) and realtime (#107)

## Clarifying addendum (#131 — not a schema fork)

Status: locked as clarification of v0.1 (Cole / Product ACK 2026-08-23).
This is **not** v0.2. No new forked tables.

1. **`season_id` / `season` year** is the MLB championship season year
   (e.g. `2026` for the 2026 regular season). It is not interchangeable
   with `as_of`.
2. **`as_of`** (and artifact `as_of_date`) is the calendar cut date of the
   extract / publish (`YYYY-MM-DD`). A run with `as_of=2026-08-23` is
   defective if its published facts only cover `season≤2025`.
3. **Active season in `current/`:** while season `Y` is in progress (or
   until product marks it closed), every successful `current/` publish
   that feeds fantasy / FO live surfaces **MUST** include player-season
   facts for `Y` when SportsDataIO has them. Prior-only `current/` is a
   defect, not an acceptable fallback.
4. **Default window:** emitters and default dashboard filters use
   `season_year ∈ [Y-2, Y]` where `Y` is the year of `as_of_date`
   (2024–2026 when `Y=2026`). Derive the window; do not hardcode those
   years forever. `SPORTSDATAIO_SEASONS` / `sportsdataio.seasons` may
   override the extract list.
5. **Live path:** SportsDataIO `player_season_stat` (and `player_game_stat`
   rollups when the season stub is thin) is the live path for in-season
   years Lahman does not have. Overlay those years onto
   `player_season_metrics` / `current/fantasy/cards.jsonl`. Keep Lahman +
   BR rWAR historical rows intact. Do not dual-write WAR onto Stats API
   or SDIO spine facts.
6. **Soft-fail:** missing `SPORTSDATAIO_API_KEY` still exits 0 and skips
   the spine. That path must surface a clear signal (`extract_report`
   `current_season_missing`, `metrics_manifest.json`, and a logged
   warning). It must not silently ship prior-only metrics as if the
   current season were present.
