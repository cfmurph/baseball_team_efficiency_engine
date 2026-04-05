# Baseball Team Efficiency Engine

A production-style baseball analytics project that expands a simple team/payroll merge into a reusable analytics platform for MLB team efficiency, roster construction, and performance modeling.

## What this project adds

- Multi-stage data pipeline with extract, transform, and load separation
- DuckDB warehouse with fact/dimension tables
- dbt-ready transformation layer
- Advanced team efficiency metrics and salary concentration measures
- Modeling scaffolding for win prediction, clustering, and efficiency frontier analysis
- Streamlit dashboard starter for team comparison and trend analysis
- Data quality checks and test scaffolding
- Clear roadmap for adding WAR, Statcast, transactions, and injuries

## Architecture

```text
Raw CSV/API data
    -> pipeline/extract
    -> pipeline/transform
    -> DuckDB warehouse
    -> dbt marts
    -> models / dashboard / exports
```

## Repo layout

```text
config/                 YAML configuration
pipeline/               ETL entrypoints
src/baseball_analytics/ Reusable Python package
models/                 Predictive + clustering models
dbt/                    dbt project scaffold
dashboard/              Streamlit app
docs/                   Product, schema, roadmap, metrics docs
tests/                  Unit + data quality tests
artifacts/              Output tables and plots
```

## Core outputs

- `team_onfield_contract_metrics.csv`
- `team_efficiency_frontier.csv`
- `team_clusters.csv`
- `win_projection_features.csv`
- DuckDB warehouse: `baseball_analytics.duckdb`

## Quick start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run full pipeline

```bash
python -m pipeline.extract.pull_sources
python -m pipeline.transform.build_warehouse
python -m pipeline.transform.build_metrics
python -m models.train_win_model
python -m models.cluster_teams
```

### 3) Launch dashboard

```bash
streamlit run dashboard/app.py
```

## Current data sources

Default implementation uses Lahman-derived CSV endpoints:
- Teams
- Salaries
- People
- Batting
- Pitching
- Fielding

The project is intentionally structured so WAR, Statcast, Retrosheet, transactions, or injury feeds can be added with minimal redesign.

## High-value next additions

1. Add player WAR and cost-per-WAR.
2. Add BaseRuns and roster age curves.
3. Add game-level fact table and rolling windows.
4. Add surplus-value and dead-money models.
5. Add scenario simulation for payroll redistribution.

## Notes

This repo is a strong starter framework, not a finished live MLB data product. Some advanced sources require licensing, scraping constraints, or API integration work.
