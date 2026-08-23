# ADR 0003 — MLB Stats API ingest (stub)

- Status: Proposed
- Date: 2026-08-23
- Issue: #108

## Context

After the shared lake (#105 / ADR 0001), front-office ingest switches from
thin/approximate pulls to the public MLB Stats API (no API key) for majors
team, player, game, and season data. Fantasy Phase 0 (#95) runs in parallel
on the same artifacts.

## Direction (locked on #108)

- Raw landing: `{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/…json`
  or local `data/raw/mlb_stats/…`. Re-pull of the same date is idempotent.
- Warehouse: new/extended facts joined via MLB ids; Lahman people/team keys
  remain the historical bridge.
- Do **not** replace Lahman salaries, Lahman ID master, or BR rWAR.
- Nightly: add the extract after `pull_sources` (before or after `pull_war`).
  Warehouse must still build on the Lahman-only path.
- Rate-limit politely; fail soft with a validation report.

## Non-goals

Full extract implementation, #106 thin API, #107 realtime, and required
Sportradar are out of scope here. Implement against #108.

## Pointers

- Contract comment: https://github.com/cfmurph/baseball_team_efficiency_engine/issues/108#issuecomment-5388826516
- WAR SoT: [0002-source-of-truth-map.md](0002-source-of-truth-map.md)
- Lake layout: [0001-shared-artifact-contract.md](0001-shared-artifact-contract.md)
