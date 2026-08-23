# ADR 0001 — Shared artifact lake layout

Status: Accepted (lands with #105; no fantasy pipeline in that slice)

## Context

Nightly outputs must be one shared, versioned set under a vendor-agnostic
`ARTIFACTS_URI` so every dashboard reads the same files. A later Fantasy
Phase 0 job will emit a card payload on the same lake. #105 must not
implement those cards — only leave a place for them.

## Decision

Keep the #105 prefix and partition. Do **not** introduce a second URI,
bucket, or versioning scheme for new product files.

```text
{ARTIFACTS_URI}/
  {league}/{level}/{run_date}/
    manifest.json
    <flat metric CSVs and plots>      # today's publish
    <any extra relative path>         # first-class; no layout change
  {league}/{level}/latest/            # copy of the newest successful run
    …same tree…
```

- **URI / credentials** stay `ARTIFACTS_URI` + env (S3-compatible).
- **Partition** stays `{league}/{level}/{run_date|latest}` (defaults `mlb/mlb`).
- **Object key** is `{partition}/{relative path from artifacts/}`.
- Nightly already walks `artifacts/` recursively (skips `.remote_cache/`).
  Adding a file or subdirectory is enough; upload and `manifest.json`
  pick it up.
- Resolvers keep the relative path (not only the basename) so a nested
  key loads the same way as a top-level CSV.

`latest/` is overwritten only after a fully successful pipeline + upload.
Dated prefixes are immutable for that calendar day.

## Reserved path (not written by #105)

| Relative key | Owner | When |
|---|---|---|
| `fantasy/cards.jsonl` | Fantasy Phase 0 follow-up | After #105 |

A later job writes `artifacts/fantasy/cards.jsonl` (JSONL; schema owned by
that follow-up). Nightly publish then stores:

```text
{uri}/{league}/{level}/{run_date}/fantasy/cards.jsonl
{uri}/{league}/{level}/latest/fantasy/cards.jsonl
```

No new `ARTIFACTS_*` variable, no new partition token, no API. Card
schema, marketing fields, and dashboard fantasy UI are out of scope here.

## Consequences

- Extra published files do not force a lake redesign.
- Dashboard #105 loaders stay on existing metric CSV names.
- A consumer that wants cards asks for `fantasy/cards.jsonl` the same way
  it asks for `team_onfield_contract_metrics.csv`.
- `runs/{run_id}/` + `current/` from the earlier architect sketch is
  equivalent in spirit (`run_date` + `latest`). Do not fork layouts
  unless a second client needs a different deploy cadence.
