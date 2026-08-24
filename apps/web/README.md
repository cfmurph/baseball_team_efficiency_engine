# BenchOrStart (`apps/web`)

Public Next.js client for BenchOrStart. Consumes the thin `#106` / `#152` read API (`GET /v1/health`, `/v1/cards`, `/v1/seasons`, `/v1/players`, `/v1/players/{id}`). `/` is cards. `/players` hits live players routes when `NEXT_PUBLIC_API_URL` is set. Unset URL keeps fixture cards and an empty players miss — no invented 2026 rows.

Routes: `/` cards home, `/players` directory, `/players/[id]` player page (`id` = schema 1.0 `player.player_id`). No `/compare` in this wave.

The Streamlit shell at `dashboard/fantasy_app.py` stays as a local fallback. The front-office GM app (`dashboard/app.py`) is out of scope.

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
| `NEXT_PUBLIC_API_URL` | `#106` origin. Unset → fixture cards + stub health. |
| `NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING` | QA: raise the not-current-year banner on stubs. |
| `FANTASY_WAITLIST_WEBHOOK` | Server-only HTTPS POST `{email,source,created_at}`. |
| `FANTASY_WAITLIST_PATH` | Optional JSONL append (local). No-op success on Vercel if the disk is read-only. |

## Vercel

Create a project on this repo:

1. **Root Directory:** `apps/web` (include files outside the root so `packages/*` resolve).
2. Framework: Next.js (`next.config.ts` already lists `transpilePackages`).
3. Env: `NEXT_PUBLIC_API_URL` when the API is live. Optional waitlist webhook. No FO secrets.

## Copy lock

Headline, subhead, CTA, waitlist, footer, invite chip, and share chrome come from `@bos/card-schema` and must match `fantasy/copy.py` until marketing revises them.
