# Baseball Team Efficiency Engine

A production-grade MLB analytics platform that turns Lahman baseball data into a full analytical warehouse with advanced metrics, predictive modeling, and an interactive dashboard.

## What this project does

- **Star-schema DuckDB warehouse** — fact/dimension tables for teams, players, seasons, and salaries
- **Player-level WAR** — Baseball-Reference rWAR (1871–present) joined onto Lahman player-seasons; wOBA/FIP approximations remain as fallback (`war_source = real|approx`)
- **Advanced team metrics** — BaseRuns, Pythagorean gap, cost-per-WAR, surplus value, dead money, Gini coefficient, window phase detection
- **Contract efficiency analysis** — classify every player contract as surplus value, fair value, overpaid, or dead money
- **Two-model win prediction** — Linear Regression baseline + XGBoost with feature importance
- **Efficiency frontier** — polynomial envelope of payroll vs wins; teams above curve are efficient
- **Team clustering** — KMeans archetypes: Big-Spend Contender, Low-Spend Overachiever, Rebuilding, Declining Spender
- **Data validation** — lightweight but thorough checks on every pipeline stage
- **8-section Streamlit dashboard** — Overview, Team Deep Dive, Compare Teams, Roster Lab, Contract Watch, Efficiency Frontier, What-If Sim, Model Insights (works with missing artifacts via empty states)

## Architecture

```text
Raw CSV / Lahman API
    → pipeline/extract/pull_sources.py
    → pipeline/extract/pull_war.py            (Baseball-Reference rWAR)
    → pipeline/transform/build_warehouse.py   (DuckDB star schema + validation)
    → pipeline/transform/build_metrics.py     (CSV artifacts per analysis)
    → models/train_win_model.py               (LinearRegression + XGBoost + frontier)
    → models/cluster_teams.py                 (KMeans team archetypes)
    → dashboard/app.py                        (Streamlit 8-section FO / GM UI)
    → dashboard/fantasy_app.py                (BenchOrStart waitlist + share cards)
```

## Repo layout

```text
config/                         YAML configuration (sources, modeling knobs, WAR constants)
pipeline/
  extract/pull_sources.py       Download Lahman CSVs
  extract/pull_war.py           Download Baseball-Reference rWAR (optional; approx fallback)
  transform/build_warehouse.py  Build star schema + WAR + metrics + validation
  transform/build_metrics.py    Export CSVs for team, player, contract analysis
src/baseball_analytics/
  config.py                     Settings loader + ARTIFACTS_URI resolution
  storage.py                    Shared lake upload (runs/ + current/) + local fallback
  fantasy.py                    Phase 0 fantasy/cards.jsonl stub
  io.py                         CSV I/O helpers
  metrics.py                    All metric functions (Pythag, Gini, WAR efficiency, contract labels)
  war.py                        rWAR overlay + Lahman approx (wOBA / FIP) + BaseRuns
  schema.py                     DuckDB DDL for all fact/dim tables
  validation.py                 Data quality checks + ValidationReport
models/
  train_win_model.py            LinearRegression + XGBoost win models + efficiency frontier
  cluster_teams.py              KMeans team archetype clustering
dbt/                            dbt scaffold (staging + mart SQL models)
dashboard/app.py                Streamlit multi-section FO / GM dashboard
dashboard/fantasy_app.py        BenchOrStart waitlist + share-card shell
fantasy/                        Card loader, stub cards.jsonl, waitlist helper
docs/                           Architecture, ADRs, schema, metrics framework, shared artifacts, roadmap
tests/                          Unit tests covering metrics, WAR, validation, artifact storage
artifacts/                      Output CSVs and plots (gitignored, generated at runtime)
```

## Warehouse schema

### Dimensions
| Table | Key columns |
|---|---|
| `dim_team` | team_key, team_id, franchise_id, team_name, league_id |
| `dim_season` | season_key, year_id |
| `dim_player` | player_id, name_full, birth_year, bats, throws |

### Facts
| Table | Grain | Key new columns |
|---|---|---|
| `fact_team_season` | team × season | team_total_war, war_source, cost_per_war, surplus_value, dead_money_share, base_runs, window_phase |
| `fact_player_season` | player × season × team | batting_war, pitching_war, player_war, war_source, surplus_value, contract_label |
| `fact_salary` | player × season × team | salary |

## Key metrics

| Metric | Formula |
|---|---|
| Batting WAR | Baseball-Reference rWAR when mapped; else wOBA → wRAA → runs above replacement / RPW |
| Pitching WAR | Baseball-Reference rWAR when mapped; else FIP → runs prevented vs league avg RA/9 |
| WAR source | `real` (rWAR) or `approx` (Lahman); team grain may be `mixed` |
| Cost per WAR | payroll / team_total_war |
| Surplus value (team) | (WAR × $8M/WAR) − payroll |
| Surplus value (player) | (player_war × $8M/WAR) − salary |
| Dead money share | payroll tied to players with WAR < 0.5 |
| Pythagorean gap | actual wins − Pythagorean expected wins |
| BaseRuns gap | actual runs scored − BaseRuns estimate |
| Window phase | contending / developing / rebuilding / declining / steady |

## Artifacts generated

```
team_onfield_contract_metrics.csv    All team-season metrics
team_efficiency_frontier.csv         With efficiency_label (low/below_avg/above_avg/elite)
team_window_phases.csv               Latest phase per franchise
team_clusters.csv                    KMeans archetypes per team-season
team_cluster_summary.csv             Mean stats per archetype
player_season_metrics.csv            All player-season metrics (Phase 0 aliases + ranks)
player_top_surplus_value.csv         Best-value players
player_worst_contracts.csv           Most negative surplus value
player_dead_money.csv                Players with WAR ≤ 0 and salary > 0
win_model_metrics.csv                MAE + R² for both models
win_model_predictions.csv            Actual vs predicted + error per team-season
win_model_feature_importance.csv     XGBoost feature importances
win_model_frontier_data.csv          Above/below efficiency frontier per team-season
win_model_actual_vs_predicted.png    Side-by-side LR vs XGBoost scatter
win_model_efficiency_frontier.png    Payroll vs wins with polynomial frontier
team_clusters_scatter.png            Cluster scatter by archetype
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (runs in ~2–3 minutes)
python3 -m pipeline.extract.pull_sources
python3 -m pipeline.extract.pull_war
python3 -m pipeline.transform.build_warehouse
python3 -m pipeline.transform.build_metrics
python3 -m models.train_win_model
python3 -m models.cluster_teams

# Dashboard (run from the repo root)
streamlit run dashboard/app.py --server.port 8501 --server.headless true
```

The dashboard is Streamlit + Plotly only. Pages load CSVs through named helpers in `dashboard/data.py` (`load_team_metrics()`, `load_player_season_metrics()`, …) — never raw `Path("artifacts")`. When `ARTIFACTS_URI` is set the loaders use `resolve_artifact()` (`current/`, then a one-release `latest/` compat bridge, then local `artifacts/`). Sidebar **Source** shows `remote` | `local` | `missing`. See [docs/adr/0001-shared-artifact-contract.md](docs/adr/0001-shared-artifact-contract.md) and [docs/shared_artifacts.md](docs/shared_artifacts.md).

Season, team, and league widgets share `st.session_state` keys `season_year`, `selected_team`, and `selected_league` (documented in `dashboard/state.py`) so a pick on Overview carries to Team Deep Dive.

## Nightly refresh

The six pipeline steps above are also wrapped by a fail-fast orchestrator:

```bash
python3 -m pipeline.run_nightly
# optional: --config-path config/settings.yaml
```

That command runs extract → rWAR extract → warehouse → metrics → win model → clustering in order, logs timing for each step, and **stops on the first non-zero exit** (later steps are named in the error and are not run). Use the same command locally whenever you want a full refresh.

GitHub Actions runs it overnight via `.github/workflows/nightly-refresh.yml`:

- **Schedule:** `0 8 * * *` UTC = **2:00 AM America/Edmonton during MDT** (UTC-6). During MST (UTC-7) that is 1:00 AM local. Actions cron is UTC-only and cannot follow DST.
- **Manual trigger:** Actions → **Nightly data refresh** → **Run workflow** (`workflow_dispatch`).
- **Outputs:** CSVs, plots, and the DuckDB warehouse stay gitignored. The workflow uploads them as the `nightly-artifacts` run artifact (14-day retention) instead of committing generated files.
- **Shared storage (optional):** when `ARTIFACTS_URI` is set, the orchestrator uploads `artifacts/` to immutable `runs/{run_id}/` and promotes `current/` only after a full success. The dashboard reads `current/` and falls back to local `artifacts/` if the URI is unset or unreachable. Source badge: `remote` | `local` | `missing`. See [docs/adr/0001-shared-artifact-contract.md](docs/adr/0001-shared-artifact-contract.md) and [docs/shared_artifacts.md](docs/shared_artifacts.md).

Optional Sportradar pulls are not part of this job; they still require `SPORTRADAR_API_KEY` and `python3 -m pipeline.extract.pull_sportradar`.

### Shared artifact storage

```bash
# AWS S3
export ARTIFACTS_URI=s3://my-bucket/baseball-analytics
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Cloudflare R2 — r2:// or s3:// URI, plus the account endpoint
export ARTIFACTS_URI=r2://my-r2-bucket/baseball-analytics
export AWS_ENDPOINT_URL=https://<accountid>.r2.cloudflarestorage.com
export AWS_DEFAULT_REGION=auto

# CI / QA filesystem
export ARTIFACTS_URI=file:///tmp/btee-qa
```

Leave `ARTIFACTS_URI` empty for local-only. GitHub Actions reads these from repository secrets (never commit keys). Full variable list and QA steps: [docs/shared_artifacts.md](docs/shared_artifacts.md).

## Dashboard sections

Product nav is grouped in the sidebar (League / Roster / Models). Entrypoint is still `streamlit run dashboard/app.py`.

1. **Overview** — Command-center KPIs, surplus leaderboards, payroll-vs-wins scatter, ranking, standings, and window phases
2. **Team Deep Dive** — Franchise dossier: KPIs, history, trajectory charts, and season roster
3. **Compare Teams** — Multi-team table + metric trends across a year range
4. **Roster Lab** — Player WAR vs salary scatter with contract classification
5. **Contract Watch** — Surplus value / overpaid / dead money / fair value tables
6. **Efficiency Frontier** — Teams above/below polynomial payroll-wins envelope + cluster archetypes
7. **What-If Sim** — Estimated win change from a payroll increase
8. **Model Insights** — Feature importance, actual vs predicted, largest model misses

Each section stays usable when its CSV is missing: the UI shows a short empty state and the pipeline command to generate it. Lahman payroll typically ends in 2016 — recent seasons may show standings without dollar metrics. This app is the front-office GM dashboard (no fantasy start/sit, waitlist, or share cards).

## Running tests

```bash
python3 -m pytest tests/ -v
```

Unit tests covering: metrics helpers, approximate WAR, Baseball-Reference rWAR overlay + ID mapping, BaseRuns, contract classification, window detection, data validation checks.

CI smokes on PRs to `master` (`.github/workflows/ci-smoke.yml`):

```bash
python3 -m pytest tests/test_dashboard_apptest.py tests/test_run_nightly.py tests/test_golden_war.py -v
```

- **AppTest** — every sidebar page boots without exception (empty `artifacts/` is fine).
- **Nightly contract** — `pull_war` stays in `PIPELINE_STEPS` immediately after `pull_sources`.
- **Golden WAR** — Judge 2022, Trout 2012, deGrom 2018, Ohtani 2023 stay `war_source=real` against committed fixtures. Refresh notes: [docs/war_sources.md](docs/war_sources.md#golden-fixtures-ci).

## Data sources

- **Lahman Baseball Database** via the [Rdatasets CDN](https://vincentarelbundock.github.io/Rdatasets/csv/Lahman/) — teams, people, batting, pitching, salaries (salary coverage ends ~2016).
- **Baseball-Reference rWAR** — [war_daily_bat.txt](https://www.baseball-reference.com/data/war_daily_bat.txt) and [war_daily_pitch.txt](https://www.baseball-reference.com/data/war_daily_pitch.txt), 1871–present. Player IDs join through Lahman `People.bbrefID`; team IDs through `data/crosswalks/br_team_map.csv`. See [docs/war_sources.md](docs/war_sources.md).

Lahman extract (`pull_sources`) and WAR extract (`pull_war`) are separate. The warehouse builds without rWAR files and marks every row `war_source=approx`.

## High-value next additions

1. ~~Pull real WAR from Baseball Reference or FanGraphs API.~~ Shipped: Baseball-Reference rWAR (`pipeline.extract.pull_war`).
2. Add `fact_game` table from Retrosheet for bullpen/clutch analysis.
3. Add roster transaction log for trade analysis.
4. Monte Carlo wins simulation for payroll redistribution.
5. Add Prefect/Dagster orchestration for scheduled refresh.
6. Containerize with Docker + Postgres for shared deployment.
