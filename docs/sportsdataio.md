# SportsDataIO ingest (Phase 0 / schema v0.1)

Primary live ingest for schema v0.1. Locked contract:
[docs/architecture/phase0-schema-v0.1.md](architecture/phase0-schema-v0.1.md).

SportsDataIO is **additive**. Lahman, Baseball-Reference rWAR, and the MLB
Stats API path from #123 stay in place.

## Auth

`SPORTSDATAIO_API_KEY` is read from the environment only. Never commit a
key. GitHub Actions nightly maps `secrets.SPORTSDATAIO_API_KEY` → env.

When the key is missing the extract **soft-fails** (exit 0), writes
`extract_report.json` with `ok: false` / `skipped_reason: missing_api_key`,
and the warehouse skips the spine. CI smoke without the secret still passes.

Nightly ingest **soft-fails** when the key is absent (CI / forks stay green).
The dedicated probe is the auth-proof path and **hard-fails**:

**Actions → SportsDataIO auth probe → Run workflow**

(`sdio-probe.yml`, `workflow_dispatch` only — not a pull_request check).
Missing secret exits 1 with `SPORTSDATAIO_API_KEY missing`. A 2xx `Teams`
response logs HTTP status plus payload shape (`len` / keys) only. Any
non-2xx exits 1 with `http_status=<code>` only. The key is never echoed.

## Raw landing

```text
{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/…json
data/raw/sportsdataio/{endpoint}/{as_of_date}/…json
```

Sibling of `runs/{run_id}/` and `current/`. Re-pull of the same date is
idempotent. `latest/` is not used for raw.

## Endpoints

| Endpoint token | HTTP path | Default file |
|---|---|---|
| `teams` | `/v3/mlb/scores/json/Teams` | `teams.json` |
| `players` | `/v3/mlb/scores/json/Players` | `players.json` |
| `games_by_date` | `/v3/mlb/scores/json/GamesByDate/{YYYY-MMM-DD}` | `games_by_date_{as_of}.json` |
| `player_game_stats` | `/v3/mlb/stats/json/PlayerGameStatsByDate/{YYYY-MMM-DD}` | `player_game_stats_{as_of}.json` |
| `player_season_stats` | `/v3/mlb/stats/json/PlayerSeasonStats/{season}` | `player_season_stats_{year}.json` |
| `games` | `/v3/mlb/scores/json/Games/{season}` | `games_{year}.json` (opt-in) |
| `extract_report` | (written by us) | `extract_report.json` |

Default pull is incremental: Teams + Players bootstrap, then date feeds for
`as_of_date`, plus a thin `PlayerSeasonStats` stub. `--include-season-feeds`
adds season-wide `Games/{season}`.

The client sends the key as `Ocp-Apim-Subscription-Key` (never in the URL
or landed JSON). Default interval 0.5s; 3 retries on 429 / 5xx.

## How to refresh

```bash
python3 -m pipeline.extract.pull_sportsdataio
python3 -m pipeline.extract.pull_sportsdataio --season 2024 --as-of-date 2026-08-23

export ARTIFACTS_URI=file:///tmp/btee-qa
python3 -m pipeline.extract.pull_sportsdataio
```

Nightly (`python3 -m pipeline.run_nightly`) runs this after `pull_mlb_stats`
and before `build_warehouse`. GitHub Actions nightly injects
`secrets.SPORTSDATAIO_API_KEY` but the extract still soft-fails if the
secret is empty. To prove the key: **Actions → SportsDataIO auth probe →
Run workflow** (hard-fail if missing or non-2xx).

## Warehouse spine

`build_warehouse` calls `load_sdio_frames()` after Lahman + BR rWAR + Stats
API tables. Empty raw → skip. Landed JSON becomes:

| Table | Grain | Notes |
|---|---|---|
| `player` / `team` / `game` | UUID identity | Bootstrapped from SDIO ids |
| `external_id_alias` | `(system, entity_type, external_id)` | `sportsdataio` primary; `mlb` / `bbref` / `lahman` when joined |
| `player_game_stat` | `(player_id, game_id)` | Provenance required |
| `player_season_stat` | `(player_id, season, team_id)` | Thin rollup stub |

No `fantasy_*_stat` or `scout_*_stat` tables. These facts do not overwrite
`fact_player_season.player_war`.

## Out of scope

Thin read API (#106), realtime (#107), RBAC, forked fantasy/scout stats.
`current/fantasy/cards.jsonl` and `share.stat_line` are unchanged.
