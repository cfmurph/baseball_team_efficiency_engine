# Architecture

## Goal

Turn a simple historical baseball CSV merge into an analytics platform that can answer:
- Which teams convert payroll into wins most effectively?
- Which roster constructions are fragile or concentrated?
- Which clubs overperform or underperform talent/payroll signals?
- What team archetypes exist across eras?
- How well can wins be forecast from payroll and team characteristics?

## Layers

### 1. Ingestion
- Pull raw Lahman CSVs into `data/raw` (`pull_sources`)
- Pull Baseball-Reference rWAR text files (`pull_war`) — optional; warehouse falls back to approx
- Maintain one file per source
- Preserve raw column names for traceability

### 2. Warehouse
DuckDB is used first because it is fast, lightweight, and ideal for local analytical workflows.

Tables:
- `dim_team`
- `dim_season`
- `fact_salary`
- `fact_team_season`

Future additions:
- `fact_player_season`
- `fact_game`
- `fact_transaction`
- `fact_injury`

### 3. Semantic / dbt layer
This is where metric logic becomes standardized and reusable:
- team efficiency mart
- roster concentration mart
- player value mart
- scenario simulation input mart

### 4. Modeling layer
- Win prediction model
- Team cluster model
- Efficiency frontier regression
- Future: dead-money and surplus-value models

### 5. Presentation layer
- Streamlit dashboard for season, team, and trend exploration
- Static exports for portfolio/demo use
- Later: API endpoints and scheduled refresh

## Suggested production path

### Current state
- Local batch pipeline
- Fail-fast nightly orchestrator (`python3 -m pipeline.run_nightly`)
- GitHub Actions schedule at 08:00 UTC (2:00 AM Mountain Daylight Time)
- File-based configuration

### Next production milestones
1. Add heavier orchestration (Prefect or Dagster) if the refresh outgrows Actions.
2. Add source contracts and validation checks.
3. Containerize with Docker.
4. Persist warehouse in Postgres for shared use.
5. Publish dashboard.
6. Add CI for tests and linting.
