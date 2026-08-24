# BenchOrStart (`apps/web`)

Public Next.js client for BenchOrStart. Consumes the thin **#144 / #106** read API (`GET /v1/health`, `/v1/cards`, `/v1/seasons` per `services/api/openapi.yaml`). When `NEXT_PUBLIC_API_URL` is unset it renders the four fixture cards from `fantasy/stub_cards.jsonl` — no invented live 2026 rows. Types already match the OpenAPI; swapping the env URL is the only change.

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

See `.env.example`. Only `NEXT_PUBLIC_API_URL` is required to leave stub mode. Do not put `SPORTSDATAIO_API_KEY`, lake credentials, or other FO secrets in this app.

| Variable | Role |
|---|---|
| `NEXT_PUBLIC_API_URL` | `#144` / `#106` origin. Unset → fixture cards + stub health. |

Against master `/v1` (preferred when this process is up; fixture fallback otherwise):

```bash
export ARTIFACTS_URI=file://$PWD/tests/fixtures/api/lake_current
python3 -m services.api
export NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
npm run dev
```
| `NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING` | QA: raise the #137 banner on stubs. Runtime alias: `STUB_CURRENT_SEASON_MISSING`. |
| `FANTASY_WAITLIST_WEBHOOK` | Server-only HTTPS POST `{email,source,created_at}`. |
| `FANTASY_WAITLIST_PATH` | Optional JSONL append (local). No-op success on Vercel if the disk is read-only. |

## Vercel

Create a project on this repo:

1. **Root Directory:** `apps/web` (include files outside the root so `packages/*` resolve).
2. Framework: Next.js (`next.config.ts` already lists `transpilePackages`).
3. Env: `NEXT_PUBLIC_API_URL` when the API is live. Optional waitlist webhook. No FO secrets.

## Copy lock

Headline, subhead, CTA, tabs, footer, empty states, waitlist, invite chip, and share chrome come from `@bos/card-schema` and must match `fantasy/copy.py` VERBATIM. Cole owns that file — do not rewrite.

Live 2026 rows are not a ship gate. If `/v1` only has prior years, show `PRIOR_SEASON_BANNER` and ship those cards. Keyed 2026 publish stays on #131.
