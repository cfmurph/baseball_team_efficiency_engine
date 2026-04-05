# Baseball Team Efficiency Engine

A production-grade MLB analytics platform that turns Lahman baseball data into a full analytical warehouse with advanced metrics, predictive modeling, and an interactive dashboard.

## What this project does

- **Star-schema DuckDB warehouse** — fact/dimension tables for teams, players, seasons, and salaries
- **Player-level WAR approximations** — wOBA-based batting WAR and FIP-based pitching WAR derived entirely from Lahman data
- **Advanced team metrics** — BaseRuns, Pythagorean gap, cost-per-WAR, surplus value, dead money, Gini coefficient, window phase detection
- **Contract efficiency analysis** — classify every player contract as surplus value, fair value, overpaid, or dead money
- **Two-model win prediction** — Linear Regression baseline + XGBoost with feature importance
- **Efficiency frontier** — polynomial envelope of payroll vs wins; teams above curve are efficient
- **Team clustering** — KMeans archetypes: Big-Spend Contender, Low-Spend Overachiever, Rebuilding, Declining Spender
- **Data validation** — lightweight but thorough checks on every pipeline stage
- **8-section Streamlit dashboard** — Overview, Team Deep Dive, Compare Teams, Roster Lab, Contract Watch, Efficiency Frontier, What-If Sim, Model Insights

## Architecture

```text
Raw CSV / Lahman API
    → pipeline/extract/pull_sources.py
    → pipeline/transform/build_warehouse.py   (DuckDB star schema + validation)
    → pipeline/transform/build_metrics.py     (CSV artifacts per analysis)
    → models/train_win_model.py               (LinearRegression + XGBoost + frontier)
    → models/cluster_teams.py                 (KMeans team archetypes)
    → dashboard/app.py                        (Streamlit 8-section UI)
```

## Repo layout

```text
config/                         YAML configuration (sources, modeling knobs, WAR constants)
pipeline/
  extract/pull_sources.py       Download Lahman CSVs
  transform/build_warehouse.py  Build star schema + WAR + metrics + validation
  transform/build_metrics.py    Export CSVs for team, player, contract analysis
src/baseball_analytics/
  config.py                     Settings loader
  io.py                         CSV I/O helpers
  metrics.py                    All metric functions (Pythag, Gini, WAR efficiency, contract labels)
  war.py                        Batting WAR (wOBA-based) + Pitching WAR (FIP-based) + BaseRuns
  schema.py                     DuckDB DDL for all fact/dim tables
  validation.py                 Data quality checks + ValidationReport
models/
  train_win_model.py            LinearRegression + XGBoost win models + efficiency frontier
  cluster_teams.py              KMeans team archetype clustering
dbt/                            dbt scaffold (staging + mart SQL models)
dashboard/app.py                Streamlit multi-section dashboard
docs/                           Architecture, schema, metrics framework, roadmap, product brief
tests/                          53 unit tests covering metrics, WAR, validation
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
| `fact_team_season` | team × season | team_total_war, cost_per_war, surplus_value, dead_money_share, base_runs, window_phase |
| `fact_player_season` | player × season × team | batting_war, pitching_war, player_war, surplus_value, contract_label |
| `fact_salary` | player × season × team | salary |

## Key metrics

| Metric | Formula |
|---|---|
| Batting WAR | wOBA → wRAA → runs above replacement / RPW |
| Pitching WAR | FIP → runs prevented vs league avg RA/9 |
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
player_season_metrics.csv            All player-season metrics
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
python3 -m pipeline.transform.build_warehouse
python3 -m pipeline.transform.build_metrics
python3 -m models.train_win_model
python3 -m models.cluster_teams

# Dashboard
streamlit run dashboard/app.py
```

## Dashboard sections

1. **Overview** — Season efficiency scatter (payroll vs wins), KPIs, ranking table
2. **Team Deep Dive** — Win trajectory, payroll, WAR, window phase timeline
3. **Compare Teams** — Multi-team line chart across any metric, any date range
4. **Roster Lab** — Player WAR vs salary scatter with contract classification
5. **Contract Watch** — Top surplus value / worst contracts / dead money tables
6. **Efficiency Frontier** — Teams above/below polynomial payroll-wins envelope + cluster archetypes
7. **What-If Sim** — Estimated win change from payroll increase
8. **Model Insights** — Feature importance, actual vs predicted, largest model misses

## Running tests

```bash
python3 -m pytest tests/ -v
```

53 tests covering: metrics helpers, WAR computation, BaseRuns, contract classification, window detection, data validation checks.

## Data sources

All data from [Lahman Baseball Database](https://www.seanlahman.com/baseball-archive/statistics/) via the `Rdatasets` CDN.

The project is structured so WAR from FanGraphs/Baseball Savant, Statcast, Retrosheet game logs, or injury data can be added with minimal redesign — just add a new source URL in `config/settings.yaml` and a corresponding loader in `build_warehouse.py`.

## High-value next additions

1. Pull real WAR from Baseball Reference or FanGraphs API.
2. Add `fact_game` table from Retrosheet for bullpen/clutch analysis.
3. Add roster transaction log for trade analysis.
4. Monte Carlo wins simulation for payroll redistribution.
5. Add Prefect/Dagster orchestration for scheduled refresh.
6. Containerize with Docker + Postgres for shared deployment.
