# ADR 0004 — SportsDataIO Phase 0 ingest (schema v0.1)

- Status: Accepted
- Date: 2026-08-23
- Issue: #128
- Schema: [../architecture/phase0-schema-v0.1.md](../architecture/phase0-schema-v0.1.md)

## Context

Cole locked Phase 0 schema v0.1 on 2026-08-23. SportsDataIO is the primary
live ingest. MLB Stats API (#123), Lahman, and BR rWAR remain. Fantasy and
scout products share one spine — no forked `fantasy_*_stat` / `scout_*_stat`
tables.

## Decision

- Raw landing: `{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/`
  or local `data/raw/sportsdataio/`. Re-pull of the same date is idempotent.
- Raw is a sibling of `runs/{run_id}/` and `current/`.
- Internal PKs are UUID5 from SDIO ids. Live joins are `external_id_alias`
  rows with `system` in `sportsdataio | mlb | bbref | fangraphs | lahman`.
- Spine grain: `player_game_stat` PK `(player_id, game_id)` plus a thin
  `player_season_stat` stub. Provenance columns are required.
- Nightly: `pull_sportsdataio` after `pull_mlb_stats`. Soft-fail (exit 0)
  when `SPORTSDATAIO_API_KEY` is missing or an endpoint blips. Warehouse
  skips the spine when raw is empty.
- Actions nightly maps `secrets.SPORTSDATAIO_API_KEY` → env.
- Auth proof: Actions → **SportsDataIO auth probe** → **Run workflow**
  (`sdio-probe.yml`, `workflow_dispatch` only). Hard-fails if the secret
  is missing or the probe endpoint is not 2xx. Nightly ingest remains
  soft-fail without the key.

## Non-goals

Removing BR/Lahman/Stats API, forked fantasy/scout stat tables, changing
`current/fantasy/cards.jsonl`, thin API (#106), realtime (#107), RBAC.
