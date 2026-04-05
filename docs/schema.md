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

### fact_salary
Grain: one row per player-team-season salary record

## Planned facts

### fact_player_season
Grain: one row per player-team-season

Planned columns:
- WAR
- PA / IP
- offensive and defensive value
- role indicators
- age

### fact_game
Grain: one row per game-team

Planned columns:
- runs scored / allowed
- starter indicators
- leverage / bullpen usage
- home-away
- opponent
