# MLB Stats API ingest

Public majors feeds from [statsapi.mlb.com](https://statsapi.mlb.com) (no API
key). Lands versioned raw, then optional warehouse tables. Baseball-Reference
rWAR (`pull_war`) stays the WAR source of truth. Lahman remains the
ID / salary bridge.

Locked contract: [ADR 0003](adr/0003-mlb-stats-api-ingest.md),
[ADR 0002](adr/0002-source-of-truth-map.md),
[ADR 0001](adr/0001-shared-artifact-contract.md).

## Raw landing

```text
{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/…json
data/raw/mlb_stats/{endpoint}/{as_of_date}/…json
```

This is a sibling of `runs/{run_id}/` and `current/`, not a second layout.
Re-pulling the same `as_of_date` overwrites that date partition only
(idempotent). `latest/` is not used for raw.

When `ARTIFACTS_URI` is unset, only the local `data/raw/mlb_stats/` tree is
written. `ARTIFACTS_URI=file:///tmp/btee-qa` writes both local and
`/tmp/btee-qa/raw/mlb_stats/…`.

## Endpoints (majors, `sportId=1`)

| Endpoint token | HTTP path | Default file |
|---|---|---|
| `teams` | `/api/v1/teams?sportId=1` | `teams.json` |
| `standings` | `/api/v1/standings?leagueId=103,104&season={year}` | `standings_{year}.json` |
| `team_hitting` | `/api/v1/teams/stats?group=hitting&stats=season` | `team_hitting_{year}.json` |
| `team_pitching` | `/api/v1/teams/stats?group=pitching&stats=season` | `team_pitching_{year}.json` |
| `player_hitting` | `/api/v1/stats?group=hitting&stats=season&playerPool=all` | `player_hitting_{year}.json` |
| `player_pitching` | `/api/v1/stats?group=pitching&stats=season&playerPool=all` | `player_pitching_{year}.json` |
| `schedule` | `/api/v1/schedule?sportId=1&season={year}&gameTypes=R` | `schedule_{year}.json` |
| `extract_report` | (written by us) | `extract_report.json` |

Default season is the year of `as_of_date` (`ARTIFACTS_AS_OF_DATE` or UTC
today). Override with `--season` / `MLB_STATS_SEASONS=2024,2025` /
`mlb_stats.seasons` in `config/settings.yaml`.

## Rate limits

The Stats API is unofficial and unpublished. This client uses polite defaults:

- `User-Agent`: project identity from `src/baseball_analytics/io.py`
- 0.35s minimum interval between requests
- 3 retries with exponential backoff on 429 / 5xx
- 45s request timeout

Do not tighten the interval in nightly. There is no API key and no quota
dashboard — if the API blips, the extract **soft-fails** (exit 0) and writes
`extract_report.json` with `ok: false`. Nightly continues; the warehouse
builds Lahman-only.

## How to refresh

```bash
python3 -m pipeline.extract.pull_mlb_stats
python3 -m pipeline.extract.pull_mlb_stats --season 2024 --as-of-date 2026-08-23

# Shared filesystem (CI / QA)
export ARTIFACTS_URI=file:///tmp/btee-qa
python3 -m pipeline.extract.pull_mlb_stats
```

Nightly (`python3 -m pipeline.run_nightly`) runs this step after `pull_war`
and before `build_warehouse`.

## Warehouse loaders and joins

`build_warehouse` calls `load_mlb_frames()` after Lahman + BR rWAR tables.
Empty raw → skip (Lahman-only). Landed JSON is parsed into:

| Table | Grain | Join |
|---|---|---|
| `dim_mlb_team_map` | MLB team | `data/crosswalks/mlb_team_map.csv` → Lahman `teamID` (year-aware) |
| `dim_mlb_player_map` | MLB player | Lahman `People.mlbID` / `mlb_id` / `key_mlbam` when present |
| `fact_mlb_team_season` | team × season | `lahman_team_id` bridge |
| `fact_mlb_player_season` | player × season × team | `lahman_player_id` + `lahman_team_id` |
| `fact_mlb_game` | game | home/away Lahman team ids |

These tables have **no WAR columns**. They do not update
`fact_player_season.player_war` or `war_source`. Existing dashboard CSVs
still come from the Lahman + rWAR facts.

## Out of scope

Thin read API (#106), realtime (#107), Sportradar, and minors.