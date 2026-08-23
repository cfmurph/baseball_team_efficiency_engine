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

The shell reads published files through the same `resolve_artifact()` / `ARTIFACTS_URI` helpers as the FO dashboard. Card sources, in order:

```text
current/fantasy/cards.jsonl
fantasy/cards.jsonl
fantasy_cards_{as_of_date}.json
```

`current/fantasy/cards.jsonl` is the marketing pointer (same relative file under `runs/{run_id}/`). `fantasy_cards_*.json` is accepted if that is what the #111 emitter lands. If neither exists the waitlist still works and labeled stub cards render.

Player CSVs (`player_season_metrics.csv` and the contract exports) are resolved the same way. This shell does not change FO data-access and is not a page in the GM 8-section app.

`edge.war_source` is `bbref` or `approx` only. `approx` rows set `is_approx: true` and show an **early model** badge.

Recommendation labels (schema v1.0): `start` → START, `sit` → BENCH, `pickup` → PICK UP, `stream` → STREAM.

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

Headline, subhead, CTA, microcopy, success, and footer strings live in `fantasy/copy.py` and must stay exact until marketing revises them.
