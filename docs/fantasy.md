# BenchOrStart (fantasy Phase 0)

Thin waitlist + share-card shell. Separate from the front-office GM dashboard.

Closes the FE slice of [#112](https://github.com/cfmurph/baseball_team_efficiency_engine/issues/112). Live cards come from the #111 nightly emitter.

## How to run

From the repo root, after `source .venv/bin/activate`:

| Surface | Command |
|---|---|
| Front office (8-section GM app) | `streamlit run dashboard/app.py` |
| BenchOrStart | `streamlit run dashboard/fantasy_app.py` |

Optional ports:

```bash
streamlit run dashboard/app.py --server.port 8501 --server.headless true
streamlit run dashboard/fantasy_app.py --server.port 8502 --server.headless true
```

## Card feed

The shell reads published files through the same `resolve_artifact()` / `ARTIFACTS_URI` helpers as the FO dashboard.

**Locked path (schema 1.0):** `current/fantasy/cards.jsonl` — that is `fantasy/cards.jsonl` under the `current/` prefix.

Fallback only if that file is missing: `fantasy/cards.jsonl`, then `fantasy_cards_{as_of_date}.json`. If none exist the waitlist still works and labeled stub cards render.

JSONL schema 1.0. `as_of_date` is on each record and `manifest.json`. `edge.war_source` is `bbref` or `approx` only. The #111 nightly emitter ranks published `player_season_metrics` into top-N **start / sit / pickup / stream** cards at `fantasy/cards.jsonl` (never `fantasy_cards_*.json`). After a successful nightly the shell reads that file so it is not empty. Bundled stubs in `fantasy/stub_cards.jsonl` render only when the lake file is missing or empty. `approx` rows set `is_approx: true` and show an **early model** badge.

Recommendation labels (schema v1.0): `start` → START, `sit` → BENCH, `pickup` → PICK UP, `stream` → STREAM.

## Soft-launch UX

- **Invite-only** chip sits next to the BenchOrStart wordmark.
- Share cards render **above** the waitlist form, tabbed by **All / START / BENCH / PICK UP / STREAM**.
- When `share.headline` is empty, the card H2 is the **player name** (`player.name`). The START/BENCH/PICK UP/STREAM badge stays a separate pill — do not reuse the recommendation label as the title.
- Face copy says **edge** for `edge.vs_replacement` (for example `+1.6 edge`). The schema field name is unchanged.
- Each card has **Copy text** (league-chat blurb: decision + player + stat line + reason + as-of) and **Download image** (PNG via Pillow, already a Streamlit/matplotlib dependency).

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

Headline, subhead, CTA, microcopy, success, and footer strings live in `fantasy/copy.py` and must stay exact until marketing revises them. Soft-launch chrome (`Invite only`, `Copy text`, `Download image`) is also in that module.
