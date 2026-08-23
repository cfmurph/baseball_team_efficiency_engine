# ADR 0001 — Shared artifact contract

- Status: Accepted
- Date: 2026-08-23
- Issues: #105 (locked comment), #109 (first publish slice), this follow-up

## Decision

Nightly outputs live under a vendor-agnostic `ARTIFACTS_URI` using an immutable
run tree plus a single mutable pointer:

```text
{ARTIFACTS_URI}/
  runs/{run_id}/                 # immutable; run_id = YYYYMMDDTHHMMSSZ or GHA run id
    manifest.json                # schema_version + as_of_date + created_at + git_sha
                                 # + pipeline_steps + war_source_summary + files[]
    metrics/*.csv                # existing filenames unchanged
    models/*                     # plots and other non-CSV model outputs
    fantasy/cards.jsonl          # Phase 0 card payload (stub/empty OK)
  current/                       # overwritten only after a fully successful nightly
    manifest.json
    …same tree…
```

`#109` published `{league}/{level}/{run_date}/` plus `latest/`. New writes use
`runs/` + `current/` only. Readers still accept the old `latest/` prefix for a
brief compat window.

## Load order and badge

1. If `ARTIFACTS_URI` is set, load `current/`.
2. On any failure or timeout, fall back to local `artifacts/`.
3. If neither has data, render the empty state.

The dashboard Source badge is exactly `remote` | `local` | `missing`.

## Immutability

- Never mutate `runs/{run_id}/` after the first successful write of that id.
- Partial or failed nightlies must not update `current/`.
- A failed `current/` promote leaves the completed run in place.

## Fantasy cards

Path is `fantasy/cards.jsonl` under both `runs/{run_id}/` and `current/`.
Not `fantasy/fantasy_cards_{as_of_date}.json`. `as_of_date` and
`schema_version` belong on `manifest.json` and on each JSONL record.
`war_source` on records is `bbref` | `approx`.

## Interface

Storage is a put/get protocol (`FileBackend`, S3-compatible `S3Backend`).
No bucket vendor is hardcoded. Accepted schemes: `s3://`, `r2://`, `gs://`,
`file://`. `file://` must work in CI without cloud credentials.

## Consequences

- Dashboards share one lake instead of per-machine `artifacts/`.
- Fantasy Phase 0 (#95) and MLB Stats API ingest (#108) consume the same tree.
- Thin API (#106) and realtime (#107) stay out of scope.
