# BenchOrStart (`apps/web`)

Public Next.js client for BenchOrStart. Consumes the thin `#106` / `#152` read API (`GET /v1/health`, `/v1/cards`, `/v1/seasons`, `/v1/players`, `/v1/players/{id}`). `/` is cards. `/players` hits live players routes when `NEXT_PUBLIC_API_URL` is set. Unset URL keeps fixture cards and an empty players miss — no invented 2026 rows.

Routes: `/` cards home, `/players` directory, `/players/[id]` player page (`id` = schema 1.0 `player.player_id`), `/compare` player compare (`ids` = the same `player_id`). Teams nav is visible but disabled.

Local boot is already **signed in** as the demo user `demo@benchorstart.local`. That is a **local mock session** (browser `localStorage` only). There is no Clerk, Auth.js, magic link, password, or JWT user table. Log out / Log in only flip that flag. Real login is parked on [#158](https://github.com/cfmurph/baseball_team_efficiency_engine/issues/158). CI does not need Clerk keys.

Waitlist is not the Next.js CTA. `WaitlistForm` / `POST /api/waitlist` stay in the tree for the Streamlit fallback and are not shown on `/`.

The Streamlit shell at `dashboard/fantasy_app.py` stays as a local fallback. The front-office GM app (`dashboard/app.py`) is out of scope. The Python `/v1` API stays anonymous.

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
| `FANTASY_WAITLIST_WEBHOOK` | Server-only HTTPS POST `{email,source,created_at}`. Unused by the Next.js CTA. |
| `FANTASY_WAITLIST_PATH` | Optional JSONL append (local). No-op success on Vercel if the disk is read-only. |

No Clerk / Auth.js env vars. Session chrome is the local mock described above.

## Vercel

Create a project on this repo:

1. **Root Directory:** `apps/web` (include files outside the root so `packages/*` resolve).
2. Framework: Next.js (`next.config.ts` already lists `transpilePackages`).
3. Env: `NEXT_PUBLIC_API_URL` when the API is live. Optional waitlist webhook (not the product CTA). No FO secrets and no Clerk keys.

## Copy lock

Headline, subhead, CTA, waitlist, footer, invite chip, and share chrome come from `@bos/card-schema` and must match `fantasy/copy.py` until marketing revises them.
