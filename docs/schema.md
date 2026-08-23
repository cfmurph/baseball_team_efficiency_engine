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

## Planned facts

### fact_game
Grain: one row per game-team

Planned columns:
- runs scored / allowed
- starter indicators
- leverage / bullpen usage
- home-away
- opponent
