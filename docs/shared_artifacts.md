# Shared artifact storage

Nightly pipeline outputs live in object storage so every dashboard instance reads the same CSVs. This slice is **S3-compatible** (AWS S3 and Cloudflare R2) with a local `artifacts/` fallback. There is no HTTP API.

## Partition layout

Under the configured URI prefix:

```text
{league}/{level}/{run_date}/<relative-file>
{league}/{level}/{run_date}/manifest.json
{league}/{level}/latest/<relative-file>
{league}/{level}/latest/manifest.json
```

`<relative-file>` is any path under local `artifacts/` (not only the top
level). New product files are added beside today's CSVs — same URI and
partition. See [ADR 0001](adr/0001-shared-artifact-layout.md).

Reserved for a Fantasy Phase 0 follow-up (not emitted by #105):

```text
{league}/{level}/{run_date}/fantasy/cards.jsonl
{league}/{level}/latest/fantasy/cards.jsonl
```

| Segment | Current default | Future example |
|---|---|---|
| `league` | `mlb` | `milb` |
| `level` | `mlb` | `aaa`, `aa`, `a+` |
| `run_date` | UTC `YYYY-MM-DD` | `2026-08-23` |
| `latest` | copy of the newest successful publish | dashboards always read this |

Example with `ARTIFACTS_URI=s3://my-bucket/baseball-analytics`:

```text
s3://my-bucket/baseball-analytics/mlb/mlb/2026-08-23/team_onfield_contract_metrics.csv
s3://my-bucket/baseball-analytics/mlb/mlb/latest/team_onfield_contract_metrics.csv
s3://my-bucket/baseball-analytics/mlb/mlb/latest/manifest.json
```

`manifest.json` lists `run_date`, `league`, `level`, `created_at`, and published filenames.

Override partition tokens with `ARTIFACTS_LEAGUE` / `ARTIFACTS_LEVEL`, or `config/settings.yaml`:

```yaml
artifacts_uri: ""          # or s3://bucket/prefix; env ARTIFACTS_URI wins
artifacts_partition:
  league: mlb
  level: mlb
```

`ARTIFACTS_RUN_DATE=YYYY-MM-DD` pins the dated partition (useful for backfills). Unset uses today's UTC date.

## Configuration

| Variable | Required | Purpose |
|---|---|---|
| `ARTIFACTS_URI` | for remote | `s3://bucket/optional-prefix` or `file:///shared/path` |
| `AWS_ACCESS_KEY_ID` | for S3/R2 | Access key (or R2 token) |
| `AWS_SECRET_ACCESS_KEY` | for S3/R2 | Secret key |
| `AWS_ENDPOINT_URL` | R2 / custom | Override the S3 API endpoint |
| `AWS_DEFAULT_REGION` | optional | Defaults to `us-east-1` (`auto` is fine for R2) |
| `ARTIFACTS_LEAGUE` | optional | Partition league (default `mlb`) |
| `ARTIFACTS_LEVEL` | optional | Partition level (default `mlb`) |
| `ARTIFACTS_RUN_DATE` | optional | Dated partition (`YYYY-MM-DD`) |
| `ARTIFACTS_CACHE_TTL` | optional | Dashboard disk-cache seconds (default `300`) |

Do **not** commit credentials. Put them in GitHub Actions secrets, a local `.env` (gitignored), or the process environment.

`ARTIFACTS_URI` overrides `artifacts_uri` in `config/settings.yaml`. When the URI is empty or unset, upload is skipped and the dashboard uses `artifacts/` only.

## AWS S3 vs Cloudflare R2

Same client (`boto3`), same `s3://` URI. The only vendor difference is the endpoint.

### AWS S3

```bash
export ARTIFACTS_URI=s3://my-bucket/baseball-analytics
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
# AWS_ENDPOINT_URL unset — boto3 uses the regional AWS endpoint
```

IAM needs `s3:PutObject` (nightly) and `s3:GetObject` (dashboard) on `arn:aws:s3:::my-bucket/baseball-analytics/*`.

### Cloudflare R2

```bash
export ARTIFACTS_URI=s3://my-r2-bucket/baseball-analytics
export AWS_ACCESS_KEY_ID=<r2_access_key_id>
export AWS_SECRET_ACCESS_KEY=<r2_secret_access_key>
export AWS_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
export AWS_DEFAULT_REGION=auto
```

Create an R2 API token with Object Read+Write on that bucket. The account-level S3 API endpoint is required; do not point `ARTIFACTS_URI` at the public `r2.dev` URL.

### Shared filesystem (dev / tests)

```bash
export ARTIFACTS_URI=file:///tmp/baseball-artifacts
```

## Nightly upload

`python3 -m pipeline.run_nightly` runs extract → warehouse → metrics → models, then publishes `artifacts/` (CSVs and plots, not `.remote_cache/`) when `ARTIFACTS_URI` is set. Upload failure after a successful pipeline exits non-zero.

GitHub Actions (`.github/workflows/nightly-refresh.yml`) forwards the env vars above from repository secrets. It still uploads a 14-day `nightly-artifacts` Actions artifact as a backup.

## Dashboard load + fallback

`dashboard/app.py` resolves each CSV in this order:

1. Fresh disk cache under `artifacts/.remote_cache/` (TTL)
2. `{league}/{level}/latest/<file>` from the configured URI
3. Stale remote cache (remote unreachable)
4. Local `artifacts/<file>`
5. Missing → empty state

The sidebar **Source** line shows `local`, `shared filesystem`, or `shared s3://bucket/…`.

BenchOrStart (`dashboard/fantasy_app.py`) reads **only** `current/fantasy/cards.jsonl` through the same `resolve_artifact()` helper (local fallback `artifacts/current/fantasy/cards.jsonl`). The same relative file lives under `runs/{run_id}/fantasy/cards.jsonl` after a nightly emit. Dated `fantasy_cards_*.json` names are retired. See [fantasy.md](fantasy.md).

## QA: remote vs local parity + fallback

1. Run the pipeline once so `artifacts/` is populated.
2. Publish to a throwaway URI (R2/S3 or `file:///tmp/btee-qa`):

   ```bash
   export ARTIFACTS_URI=file:///tmp/btee-qa
   python3 -c "
   from src.baseball_analytics.config import load_artifact_settings
   from src.baseball_analytics.storage import publish_nightly_artifacts
   print(publish_nightly_artifacts())
   "
   ```

3. **Parity:** move or rename local `artifacts/*.csv`, keep `ARTIFACTS_URI` set, start the dashboard. Metrics should match the pipeline output (sidebar Source ≠ local).
4. **Fallback:** `unset ARTIFACTS_URI` (or point it at a closed port / bogus bucket) and restore local CSVs — dashboard still loads; Source is `local` (or remote-miss → local files).
5. Compare a checksum of `team_onfield_contract_metrics.csv` from `latest/` vs the local pipeline file; they should be identical.
