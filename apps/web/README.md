# BenchOrStart (`apps/web`)

Public Next.js client for BenchOrStart. Consumes live `/v1` on master (`GET /v1/health`, `/v1/cards`, `/v1/seasons` per `services/api/openapi.yaml`). When `NEXT_PUBLIC_API_URL` is unset the loader probes `http://127.0.0.1:8000`. Fixture cards from `fantasy/stub_cards.jsonl` render only if that process is down.

The Streamlit shell at `dashboard/fantasy_app.py` stays as a local fallback. The front-office GM app (`dashboard/app.py`) is out of scope — including Contract Watch (#136). This app reuses the #137 prior-only banner from `fantasy/copy.py` when `/v1/health` says the current season is missing.

## Run locally

From the repo root:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

```bash
npm test
npm run build
```

## Env

See `.env.example`. Do not put `SPORTSDATAIO_API_KEY`, lake credentials, or other FO secrets in this app.

| Variable | Role |
|---|---|
| `NEXT_PUBLIC_API_URL` | Live `/v1` origin. Unset → probe `http://127.0.0.1:8000`; stub only if that process is down. |
| `NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING` | QA: raise the #137 banner on stubs. Runtime alias: `STUB_CURRENT_SEASON_MISSING`. |
| `FANTASY_WAITLIST_WEBHOOK` | Server-only HTTPS POST `{email,source,created_at}`. |
| `FANTASY_WAITLIST_PATH` | Optional JSONL append (local). No-op success on Vercel if the disk is read-only. |

Local `/v1`:

```bash
export ARTIFACTS_URI=file://$PWD/tests/fixtures/api/lake_current
python3 -m services.api
npm run dev
```

## Vercel

Create a project on this repo:

1. **Root Directory:** `apps/web` (include files outside the root so `packages/*` resolve).
2. Framework: Next.js (`next.config.ts` already lists `transpilePackages`).
3. Env: `NEXT_PUBLIC_API_URL` pointing at the deployed `/v1`. Optional waitlist webhook. No FO secrets.

## Copy lock

Headline, subhead, CTA, tabs, footer, empty states, waitlist, invite chip, and share chrome come from `@bos/card-schema` and must match `fantasy/copy.py` VERBATIM. Cole owns that file — do not rewrite.

Live 2026 rows are not a ship gate. If `/v1` only has prior years, show `PRIOR_SEASON_BANNER` and ship those cards. Keyed 2026 publish stays on #131.
