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

The shell reads published files through the same `resolve_artifact()` / `ARTIFACTS_URI` helpers as the FO dashboard. Card keys, in order:

```text
current/fantasy/cards.jsonl
fantasy/cards.jsonl
```

Same relative file under a dated run:

```text
runs/{run_id}/fantasy/cards.jsonl
```

`fantasy/cards.jsonl` is optional. If neither key is present the waitlist still works and the card slot shows an empty state (plus labeled sample cards). `fantasy_cards_{as_of_date}.json` is retired — do not emit or load it.

Player CSVs (`player_season_metrics.csv` and the contract exports) are resolved the same way. This shell does not change FO data-access.

`edge.war_source` is `bbref` or `approx` only. `approx` rows set `is_approx: true` and show an **early model** badge.

Local fallbacks: `artifacts/current/fantasy/cards.jsonl` or `artifacts/fantasy/cards.jsonl`.

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
