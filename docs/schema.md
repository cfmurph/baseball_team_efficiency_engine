# Warehouse Schema

## Dimensions

### dim_team
- `team_key`: surrogate season-aware key seed
- `team_id`: Lahman team id
- `franchise_id`
- `team_name`
- `league_id`

### dim_season
- `season_key`
- `year_id`

## Facts

### fact_team_season
Grain: one row per team-season

Contains:
- performance metrics
- payroll aggregates
- salary concentration metrics
- efficiency metrics
- `team_total_war` / `team_batting_war` / `team_pitching_war` rolled up from effective player WAR
- `war_source`: `real` | `approx` | `mixed`

### fact_salary
Grain: one row per player-team-season salary record

### fact_player_season
Grain: one row per player-team-season

Contains:
- `batting_war` / `pitching_war` / `player_war` (effective: BR rWAR when mapped, else Lahman approx)
- `war_source`: `real` | `approx`
- PA / IP / wOBA / FIP / ERA
- salary, surplus value, contract label

## MLB Stats API (majors)

Added by #108. Loaded only when versioned raw exists under
`data/raw/mlb_stats/` or `{ARTIFACTS_URI}/raw/mlb_stats/`. Empty raw keeps
the Lahman-only warehouse. See [mlb_stats.md](mlb_stats.md).

These tables **do not** store WAR. BR rWAR on `fact_player_season` is unchanged.

### dim_mlb_team_map
- `mlb_team_id` ↔ `lahman_team_id` (year-aware via `data/crosswalks/mlb_team_map.csv`)

### dim_mlb_player_map
- `mlb_player_id` ↔ `lahman_player_id` when People exposes `mlbID` / `mlb_id` / `key_mlbam`

### fact_mlb_team_season
Grain: team × season (standings + hitting/pitching counting stats)

### fact_mlb_player_season
Grain: player × season × team (no salary, no WAR)

### fact_mlb_game
Grain: one row per game (`game_pk`) with home/away scores and Lahman team bridges

## Planned facts

### fact_game (Retrosheet / Statcast enrichment)
Grain: one row per game-team

Planned columns beyond the Stats API schedule snapshot:
- starter indicators
- leverage / bullpen usage
- opponent-adjusted context
