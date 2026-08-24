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
- Pull MLB Stats API majors feeds (`pull_mlb_stats`) into
  `{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/` or local
  `data/raw/mlb_stats/` — soft-fail; warehouse stays Lahman-only if empty
- Pull SportsDataIO Phase 0 feeds (`pull_sportsdataio`) into
  `{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/` — default
  seasons `[Y-2, Y]` from `as_of_date`; soft-fail without
  `SPORTSDATAIO_API_KEY`; warehouse skips the spine if empty; metrics
  overlay those years onto `player_season_metrics` and
  `team_onfield_contract_metrics` so `current/` can include the active
  season on both fantasy cards and FO league desks
- Maintain one file per source
- Preserve raw column names for traceability

### 2. Warehouse
DuckDB is used first because it is fast, lightweight, and ideal for local analytical workflows.

Tables:
- `dim_team`
- `dim_season`
- `fact_salary`
- `fact_team_season`

Also (when Stats API raw is present):
- `dim_mlb_team_map` / `dim_mlb_player_map`
- `fact_mlb_team_season` / `fact_mlb_player_season` / `fact_mlb_game`

Also (when SportsDataIO raw is present; schema v0.1):
- `player` / `team` / `game` (UUID identity)
- `external_id_alias`
- `player_game_stat` / `player_season_stat`

Future additions:
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
- Streamlit dashboard for season, team, and trend exploration (FO / GM — `dashboard/app.py`)
- Public BenchOrStart: Next.js (`apps/web`) over the #106 `/v1` read API
- Shared TS client (`packages/api-client`) + schema 1.0 presenters (`packages/card-schema`); Expo planned later, not scaffolded
- Streamlit `dashboard/fantasy_app.py` remains a local fallback until Next parity
- Thin read-only HTTP API (`services/api`) over published `current/` (`/v1/health`, `/v1/cards`, `/v1/seasons`)
- Static exports for portfolio/demo use
- Next.js BenchOrStart (#140) consumes the API; FO Streamlit stays internal

## Suggested production path

### Current state
- Local batch pipeline
- Fail-fast nightly orchestrator (`python3 -m pipeline.run_nightly`)
- GitHub Actions schedule at 08:00 UTC (2:00 AM Mountain Daylight Time)
- Optional shared-lake publish (`ARTIFACTS_URI`) to `runs/{run_id}/` + `current/` with dashboard remote load + local fallback
- File-based configuration
- Dashboard pages call `dashboard.data` named loaders; ARTIFACTS_URI is a loader swap, not a page rewrite
- Thin read API (`python3 -m services.api`) over the same `current/` contract (#106)

See [shared_artifacts.md](shared_artifacts.md) and [ADR 0001](adr/0001-shared-artifact-layout.md) for the `{league}/{level}/{run_date}` layout (extra files such as a later `fantasy/cards.jsonl` use the same prefix).

### Next production milestones
1. Add heavier orchestration (Prefect or Dagster) if the refresh outgrows Actions.
2. Add source contracts and validation checks.
3. Containerize with Docker.
4. Persist warehouse in Postgres for shared use.
5. Publish dashboard.
6. Add CI for tests and linting.
