# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure Python MLB analytics platform: ETL pipeline (Lahman CSV data → DuckDB warehouse → metric CSVs), ML models (win prediction + team clustering), and an 8-section Streamlit dashboard. See `README.md` for full architecture and repo layout.

### Virtual environment

The project uses a standard Python venv at `.venv/`. Always activate it before running any commands:

```bash
source /workspace/.venv/bin/activate
```

The system Python is 3.12; `python3.12-venv` must be installed at the system level (the VM snapshot includes it).

### Key commands

| Task | Command |
|---|---|
| Install deps | `pip install -r requirements.txt` |
| Run tests | `python3 -m pytest tests/ -v` |
| Run full pipeline | See "Pipeline" section below |
| Start FO dashboard | `streamlit run dashboard/app.py --server.port 8501 --server.headless true` |
| Start BenchOrStart | `streamlit run dashboard/fantasy_app.py --server.port 8502 --server.headless true` |

### Pipeline

The pipeline must be run in order. Each step is idempotent:

```bash
python3 -m pipeline.extract.pull_sources          # downloads CSVs to data/raw/ (needs internet)
python3 -m pipeline.extract.pull_war              # Baseball-Reference rWAR (needs internet)
python3 -m pipeline.extract.pull_mlb_stats        # MLB Stats API majors (soft-fail; no API key)
python3 -m pipeline.extract.pull_sportsdataio     # SportsDataIO Phase 0 (soft-fail without SPORTSDATAIO_API_KEY)
python3 -m pipeline.transform.build_warehouse     # builds DuckDB warehouse + validates
python3 -m pipeline.transform.build_metrics       # exports metric CSVs to artifacts/
python3 -m models.train_win_model                 # trains LR + XGBoost, generates plots
python3 -m models.cluster_teams                   # KMeans team archetypes
```

### Non-obvious caveats

- **No linter configured**: The repo has no pyproject.toml, ruff.toml, .flake8, or similar. There is no lint command to run.
- **Lahman salary data stops around 2016**: Dashboard pages showing payroll-related metrics for recent years (e.g., 2024) will display `NaN`. This is a data limitation, not a bug. Use years with salary data (1990–2016) for full-featured testing. Known rWAR seasons for QA (Judge 2022 ~10.8, Trout 2012 ~10.5, deGrom 2018 ~9.4) do not need salary.
- **Sportradar integration is optional**: Requires `SPORTRADAR_API_KEY` env var. Not needed for core pipeline or dashboard functionality.
- **DuckDB is embedded**: No external database server needed. The warehouse file lives at `data/warehouse/baseball_analytics.duckdb`.
- **Streamlit reads metric CSVs, not DuckDB**: When `ARTIFACTS_URI` is set the dashboard loads `current/` from the shared lake (`runs/{run_id}/` is immutable). Otherwise it uses local `artifacts/`. Remote failures fall back to local. Source badge: `remote` | `local` | `missing`. See `docs/adr/0001-shared-artifact-contract.md`. You must run the full pipeline (or point at a published run) before starting the dashboard.
- **Fantasy cards path**: `fantasy/cards.jsonl` under `runs/{run_id}/` and `current/`. `as_of_date` and `schema_version` live on the manifest / JSONL records, not in the filename.
- **All pipeline modules use `-m` syntax**: Run as `python3 -m pipeline.extract.pull_sources`, not `python3 pipeline/extract/pull_sources.py`.
- **Real WAR is Baseball-Reference rWAR**, not the Lahman wOBA/FIP approximation. Run `python3 -m pipeline.extract.pull_war` before `build_warehouse`. Player IDs map via Lahman `People.bbrefID`; team IDs via `data/crosswalks/br_team_map.csv`. Rows without a match keep approximate WAR and `war_source=approx`. See `docs/war_sources.md`.
- **MLB Stats API** is optional enrichment (`pull_mlb_stats`). Raw lands at `{ARTIFACTS_URI}/raw/mlb_stats/{endpoint}/{as_of_date}/` or `data/raw/mlb_stats/`. Soft-fail if the API blips. Warehouse builds Lahman-only when that tree is empty. Do not replace BR rWAR. See `docs/mlb_stats.md` and `docs/adr/0003-mlb-stats-api-ingest.md`.
- **SportsDataIO** is the Phase 0 live ingest (`pull_sportsdataio`). Key is `SPORTSDATAIO_API_KEY` only (never hardcoded). Soft-fail when the key is missing so CI/nightly still pass. Raw lands at `{ARTIFACTS_URI}/raw/sportsdataio/{endpoint}/{as_of_date}/` or `data/raw/sportsdataio/`. Warehouse spine: `external_id_alias` + `player_game_stat`. Schema v0.1 is LOCKED. See `docs/architecture/phase0-schema-v0.1.md` and `docs/sportsdataio.md`.
