# Shared artifact storage

Nightly pipeline outputs live in object storage so every dashboard instance
reads the same files. The locked contract is
[ADR 0001](adr/0001-shared-artifact-contract.md). This page is the operator
guide (URI, credentials, QA).

The thin read API (`python3 -m services.api`) serves `current/` over HTTP so
web clients never talk to the lake or vendor keys. Schemes for the lake
itself stay `s3://`, `r2://`, `gs://`, `file://`. See
[services/api/openapi.yaml](../services/api/openapi.yaml).

## Layout

```text
{league}/{level}/{run_date}/<relative-file>
{league}/{level}/{run_date}/manifest.json
{league}/{level}/latest/<relative-file>
{league}/{level}/latest/manifest.json
```

`<relative-file>` is any path under local `artifacts/` (not only the top
level). New product files are added beside today's CSVs — same URI and
partition. See [ADR 0001](adr/0001-shared-artifact-layout.md).

Fantasy Phase 0 cards (#111 ranked emitter) live at the locked path only:

```text
runs/{run_id}/fantasy/cards.jsonl
current/fantasy/cards.jsonl
```

| Segment | Current default | Future example |
|---|---|---|
| `league` | `mlb` | `milb` |
| `level` | `mlb` | `aaa`, `aa`, `a+` |
| `run_date` | UTC `YYYY-MM-DD` | `2026-08-23` |
| `latest` | deprecated #109 pointer; dropped next release | readers still accept it for one release |

Example with `ARTIFACTS_URI=s3://my-bucket/baseball-analytics`:

```text
s3://my-bucket/baseball-analytics/runs/20260823T080012Z/manifest.json
s3://my-bucket/baseball-analytics/runs/20260823T080012Z/metrics/player_season_metrics.csv
s3://my-bucket/baseball-analytics/runs/20260823T080012Z/fantasy/cards.jsonl
s3://my-bucket/baseball-analytics/current/manifest.json
```

Local pipeline output stays flat under `artifacts/` (plus `artifacts/fantasy/cards.jsonl`).
The `runs/` + `current/` tree is the shared URI layout.

MLB Stats API raw (#108) is a **sibling** of that published tree, not a second
layout. See [mlb_stats.md](mlb_stats.md) and
[ADR 0003](adr/0003-mlb-stats-api-ingest.md):

```text
{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/…json
{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/…json
```

SportsDataIO Phase 0 (#128) uses the same sibling-raw pattern. See
[sportsdataio.md](sportsdataio.md).

`current/` is the published SoT. `latest/` is a deprecated one-release
read-only fallback and will be dropped next release. Stats API raw does
not use `latest/`.

## Configuration

```yaml
artifacts_uri: ""          # or s3://bucket/prefix; env ARTIFACTS_URI wins
artifacts_partition:
  league: mlb              # used only by the #109 latest/ compat bridge
  level: mlb
```

| Variable | Required | Purpose |
|---|---|---|
| `ARTIFACTS_URI` | for remote | `s3://bucket/prefix`, `r2://…`, `gs://…`, or `file:///shared/path` |
| `AWS_ACCESS_KEY_ID` | for S3/R2 | Access key (or R2 token) |
| `AWS_SECRET_ACCESS_KEY` | for S3/R2 | Secret key |
| `AWS_ENDPOINT_URL` | R2 / custom | Override the S3 API endpoint |
| `AWS_DEFAULT_REGION` | optional | Defaults to `us-east-1` (`auto` is fine for R2) |
| `ARTIFACTS_RUN_ID` | optional | Pin `run_id` (else UTC timestamp or `GITHUB_RUN_ID`) |
| `ARTIFACTS_AS_OF_DATE` | optional | Manifest / card `as_of_date` (`YYYY-MM-DD`) |
| `ARTIFACTS_RUN_DATE` | optional | Alias of `ARTIFACTS_AS_OF_DATE` |
| `ARTIFACTS_CACHE_TTL` | optional | Dashboard disk-cache seconds (default `300`) |
| `ARTIFACTS_LEAGUE` / `ARTIFACTS_LEVEL` | optional | Compat-bridge tokens (default `mlb`) |

Do **not** commit credentials. Put them in GitHub Actions secrets, a local
`.env` (gitignored), or the process environment.

`ARTIFACTS_URI` overrides `artifacts_uri` in `config/settings.yaml`. When the
URI is empty or unset, upload is skipped and the dashboard uses `artifacts/`
only.

## AWS S3 vs Cloudflare R2

Same client (`boto3`), same put/get protocol. The only vendor difference is
the endpoint.

### AWS S3

```bash
export ARTIFACTS_URI=s3://my-bucket/baseball-analytics
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# AWS_ENDPOINT_URL unset — boto3 uses the regional AWS endpoint
```

IAM needs `s3:PutObject` (nightly) and `s3:GetObject` (dashboard) on
`arn:aws:s3:::my-bucket/baseball-analytics/*`.

### Cloudflare R2

```bash
export ARTIFACTS_URI=r2://my-r2-bucket/baseball-analytics
# s3://my-r2-bucket/baseball-analytics is also accepted
export AWS_ACCESS_KEY_ID=<r2_access_key_id>
export AWS_SECRET_ACCESS_KEY=<r2_secret_access_key>
export AWS_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
export AWS_DEFAULT_REGION=auto
```

Create an R2 API token with Object Read+Write on that bucket. The
account-level S3 API endpoint is required; do not point `ARTIFACTS_URI` at
the public `r2.dev` URL.

### Shared filesystem (dev / CI / QA)

```bash
export ARTIFACTS_URI=file:///tmp/baseball-artifacts
```

`file://` is the required CI path. No AWS credentials.

## Nightly upload

`python3 -m pipeline.run_nightly` runs extract → warehouse → metrics → models,
then publishes `artifacts/` (CSVs, plots, and ranked `fantasy/cards.jsonl`; not
`.remote_cache/`) when `ARTIFACTS_URI` is set. `build_metrics` ranks
`player_season_metrics` into start|sit|pickup|stream at that path. Upload
writes `runs/{run_id}/` first; `current/` is promoted only after that tree
is complete. Upload failure after a successful pipeline exits non-zero and
does not mutate an existing run id.

GitHub Actions (`.github/workflows/nightly-refresh.yml`) forwards the env
vars above from repository secrets. It still uploads a 14-day
`nightly-artifacts` Actions artifact as a backup.

## Dashboard load + fallback

`dashboard/app.py` resolves each file in this order:

1. Fresh disk cache under `artifacts/.remote_cache/` (TTL)
2. `{ARTIFACTS_URI}/current/<metrics\|models\|fantasy>/<file>`
3. Compat (deprecated, dropped next release): `{league}/{level}/latest/<file>` from #109
4. Stale remote cache (remote unreachable)
5. Local `artifacts/<file>` (or `artifacts/fantasy/cards.jsonl`)
6. Missing → empty state

The sidebar **Source** line shows `remote`, `local`, or `missing`.

BenchOrStart (`dashboard/fantasy_app.py`) reads locked `current/fantasy/cards.jsonl` (schema 1.0), then `runs/{run_id}/fantasy/cards.jsonl` for dated runs. `as_of_date` is on the record and lake manifest. `fantasy_cards_*.json` is ignored. Missing card files show an empty state plus stubs. Player CSVs use the same #105 loaders as the FO dashboard. See [fantasy.md](fantasy.md).

## QA: how to verify with `ARTIFACTS_URI=file://`

These steps are also encoded in `tests/test_storage.py`
(`test_file_uri_qa_layout_and_fallback`).

1. Run the pipeline once so `artifacts/` is populated, **or** copy a few
   CSVs into `artifacts/` for a dry run.

2. Publish to a throwaway filesystem URI:

   ```bash
   export ARTIFACTS_URI=file:///tmp/btee-qa
   export ARTIFACTS_RUN_ID=qa20260823T000000Z
   python3 -c "
   from src.baseball_analytics.config import load_artifact_settings
   from src.baseball_analytics.storage import publish_nightly_artifacts
   print(publish_nightly_artifacts())
   "
   ```

3. Confirm the locked tree (not `latest/`, not a dated cards filename):

   ```bash
   test -f /tmp/btee-qa/runs/qa20260823T000000Z/manifest.json
   test -f /tmp/btee-qa/runs/qa20260823T000000Z/fantasy/cards.jsonl
   test -f /tmp/btee-qa/current/manifest.json
   test -f /tmp/btee-qa/current/fantasy/cards.jsonl
   python3 -c "
   import json
   m=json.load(open('/tmp/btee-qa/current/manifest.json'))
   assert {'schema_version','as_of_date','created_at','git_sha','pipeline_steps','war_source_summary','files'} <= set(m)
   print(m['schema_version'], m['as_of_date'], m['files'][:5])
   "
   ```

4. **Parity:** move or rename local `artifacts/*.csv`, keep `ARTIFACTS_URI`
   set, start the dashboard. Metrics should match the pipeline output.
   Sidebar Source = `remote`.

5. **Fallback:** `unset ARTIFACTS_URI` (or point it at a closed port / bogus
   bucket) and restore local CSVs — dashboard still loads; Source is
   `local` (or `missing` if the files are gone).

6. Compare a checksum of `team_onfield_contract_metrics.csv` from
   `current/metrics/` vs the local pipeline file; they should be identical.
