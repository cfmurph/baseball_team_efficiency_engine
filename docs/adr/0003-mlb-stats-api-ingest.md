# ADR 0003 — MLB Stats API ingest

- Status: Accepted
- Date: 2026-08-23
- Issue: #108

## Context

After the shared lake (#105 / ADR 0001), front-office ingest switches from
thin/approximate pulls to the public MLB Stats API (no API key) for majors
team, player, game, and season data. Fantasy Phase 0 (#95) runs in parallel
on the same artifacts.

## Decision (locked on #108)

- Raw landing: `{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/…json`
  or local `data/raw/mlb_stats/…`. Re-pull of the same date is idempotent.
- Raw is a sibling of `runs/{run_id}/` and `current/`, not a second lake.
- Warehouse: `dim_mlb_*` + `fact_mlb_team_season` / `fact_mlb_player_season` /
  `fact_mlb_game` joined via MLB ids; Lahman people/team keys remain the
  historical bridge (`People.mlbID` when present, `mlb_team_map.csv`).
- Do **not** replace Lahman salaries, Lahman ID master, or BR rWAR
  (`pull_war` stays the WAR source of truth). Stats API tables have no WAR
  columns and do not write `fact_player_season.player_war`.
- Nightly: `pull_mlb_stats` after `pull_war`. Soft-fail (exit 0) on API
  blips. Warehouse must still build on the Lahman-only path.
- Rate-limit politely (default 0.35s); write `extract_report.json` under the
  same raw prefix.

## Non-goals

#106 thin API, #107 realtime, required Sportradar, and minors feeds.

## Pointers

- Operator guide: [../mlb_stats.md](../mlb_stats.md)
- Contract comment: https://github.com/cfmurph/baseball_team_efficiency_engine/issues/108#issuecomment-5388826516
- WAR SoT: [0002-source-of-truth-map.md](0002-source-of-truth-map.md)
- Lake layout: [0001-shared-artifact-contract.md](0001-shared-artifact-contract.md)
