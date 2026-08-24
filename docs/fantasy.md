# BenchOrStart (fantasy Phase 0)

Thin waitlist + share-card product. Separate from the front-office GM dashboard.

Public surface is **Next.js** (`apps/web`, [#140](https://github.com/cfmurph/baseball_team_efficiency_engine/issues/140)). The Streamlit shell at `dashboard/fantasy_app.py` stays as a **local fallback** until Next parity. Do not delete it. `dashboard/app.py` stays FO-only.

Live cards come from the #111 nightly emitter via the #106 `/v1` API (when present). The web client stubs that contract so it runs without the API.

## How to run

| Surface | Command |
|---|---|
| BenchOrStart (public / Next.js) | `npm install && npm run dev` → [http://localhost:3000](http://localhost:3000) |
| BenchOrStart (Streamlit fallback) | `source .venv/bin/activate && streamlit run dashboard/fantasy_app.py` |
| Front office (8-section GM app) | `source .venv/bin/activate && streamlit run dashboard/app.py` |

Optional Streamlit ports:

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
streamlit run dashboard/fantasy_app.py --server.port 8502 --server.headless true
```

See [apps/web/README.md](../apps/web/README.md) for Vercel and env.

## Card feed

The shell reads published files through the same `resolve_artifact()` / `ARTIFACTS_URI` helpers as the FO dashboard.

**Locked paths (schema 1.0):**

1. `current/fantasy/cards.jsonl` — live pointer (`fantasy/cards.jsonl` under the `current/` prefix)
2. `runs/{run_id}/fantasy/cards.jsonl` — dated run, used when `current/` is missing

`run_id` comes from `ARTIFACTS_RUN_ID` or `ARTIFACTS_RUN_DATE`, then any local `artifacts/runs/*/fantasy/cards.jsonl`.

`as_of_date` lives inside each record and the lake `manifest.json`, not in the filename. `fantasy_cards_{as_of_date}.json` is ignored.

JSONL schema 1.0. `edge.war_source` is `bbref` or `approx` only. The #111 nightly emitter ranks published `player_season_metrics` into top-N **start / sit / pickup / stream** cards at `fantasy/cards.jsonl`. After a successful nightly the shell reads `current/fantasy/cards.jsonl` so it is not empty. Bundled stubs in `fantasy/stub_cards.jsonl` render only when the lake file is missing or empty. `approx` rows set `is_approx: true` and show an **early model** badge.

Recommendation labels (schema v1.0): `start` → START, `sit` → BENCH, `pickup` → PICK UP, `stream` → STREAM.

## Soft-launch UX

- **Invite-only** chip sits next to the BenchOrStart wordmark.
- Share cards render **above** the waitlist form, tabbed by **All / START / BENCH / PICK UP / STREAM**.
- When `share.headline` is empty, the card H2 is the **player name** (`player.name`). The START/BENCH/PICK UP/STREAM badge stays a separate pill — do not reuse the recommendation label as the title.
- Face copy says **edge** for `edge.vs_replacement` (for example `+1.6 edge`). The schema field name is unchanged. Stub / sample `share.stat_line` should be empty or use **edge**. If an emitter still sends `vs repl` / `vs replacement`, render (card face, Copy text, Download image) rewrites it to **edge** or omits the line.
- Each card has **Copy text** (league-chat blurb: decision + player + stat line + reason + as-of) and **Download image** (canvas PNG in Next.js; Pillow in the Streamlit fallback).

## `/v1` client (Next.js)

`packages/api-client` talks to the #106 contract:

| Endpoint | Shape |
|---|---|
| `GET /v1/health` | `{ as_of, active_season, current_season_missing, season_window, source, seasons_present, current_season_missing_reason }` |
| `GET /v1/cards?season=&rec=` | schema 1.0 cards (`current/fantasy/cards.jsonl`) |
| `GET /v1/seasons` | `{ season_window: [Y-2, Y], seasons_present }` — window may include 2026; `seasons_present` does not invent it |

If `NEXT_PUBLIC_API_URL` is unset, the client uses the same four fixtures as `fantasy/stub_cards.jsonl` plus a health object that can set `current_season_missing` (`STUB_CURRENT_SEASON_MISSING=true` at runtime, or `NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING=true` at build). When the API is up, set the env URL — no other client change.

The web UI reuses the #137 BenchOrStart banner (`PRIOR_SEASON_BANNER` in `fantasy/copy.py`: “These picks are not the current season yet.”) when `/v1/health` says `current_season_missing` **or** max `seasons_present` is below `active_season`. Fixture stubs do not raise that banner (same as Streamlit `live_feed=False`). It does **not** invent 2026 rows. #136 Contract Watch missing-salary filtering stays FO Streamlit only — not ported here.

## QA notes

- **Cards from API or stub.** Unset `NEXT_PUBLIC_API_URL` → four stubs (Steer / Suárez / Judge / Soler) and the sample caption. Set the URL → `/v1/cards` only; empty API payloads stay empty (no silent 2026 invention).
- **Waitlist.** Email-only. Next route `POST /api/waitlist` validates, optionally POSTs `FANTASY_WAITLIST_WEBHOOK`, appends `data/waitlist/signups.jsonl` when the disk allows, otherwise no-op with the success state. Streamlit fallback uses `fantasy/waitlist.py`.
- **No `vs repl`.** Face copy, Copy text, and Download image say **edge**. Schema field `edge.vs_replacement` is unchanged.
- **Approx badge.** `war_source=approx` or `is_approx` shows the **early model** badge and hides confidence.
- **Copy lock.** `packages/card-schema` strings must match `fantasy/copy.py` (Invite only, sit→BENCH, tabs, footer, #137 prior-season banner).
- **No Contract Watch.** Missing-salary 2026 overlay rows (#136) stay on FO Streamlit. `apps/web` is cards / waitlist / share only.

## Waitlist hook (marketing)

The form is email-only. Default sink is a local JSONL file:

```text
data/waitlist/signups.jsonl
```

Override with env vars (do not commit secrets):

| Variable | Purpose |
|---|---|
| `FANTASY_WAITLIST_PATH` | Local JSONL path |
| `FANTASY_WAITLIST_WEBHOOK` | HTTPS endpoint; POST `{"email","source":"benchorstart","created_at"}` |

Point `FANTASY_WAITLIST_WEBHOOK` at Zapier, Make, Buttondown, Mailchimp, or any list API gateway. The file write still happens so QA can inspect signups without a backend.

## Copy lock

Headline, subhead, CTA, microcopy, success, and footer strings live in `fantasy/copy.py` and `packages/card-schema` and must stay exact until marketing revises them. Soft-launch chrome (`Invite only`, `Copy text`, `Download image`) is also in those modules.
