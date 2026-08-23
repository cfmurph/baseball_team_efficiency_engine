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
| Start dashboard | `streamlit run dashboard/app.py --server.port 8501 --server.headless true` |

### Pipeline

The pipeline must be run in order. Each step is idempotent:

```bash
python3 -m pipeline.extract.pull_sources          # downloads CSVs to data/raw/ (needs internet)
python3 -m pipeline.extract.pull_war              # Baseball-Reference rWAR (needs internet)
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
- **Streamlit reads metric CSVs, not DuckDB**: Pages call named loaders in `dashboard/data.py`. When `ARTIFACTS_URI` is set the dashboard loads from shared S3-compatible storage (`{league}/{level}/latest/`); otherwise it uses local `artifacts/`. Remote failures fall back to local. See `docs/shared_artifacts.md`. You must run the full pipeline (or point at a published run) before starting the dashboard.
- **All pipeline modules use `-m` syntax**: Run as `python3 -m pipeline.extract.pull_sources`, not `python3 pipeline/extract/pull_sources.py`.
- **Real WAR is Baseball-Reference rWAR**, not the Lahman wOBA/FIP approximation. Run `python3 -m pipeline.extract.pull_war` before `build_warehouse`. Player IDs map via Lahman `People.bbrefID`; team IDs via `data/crosswalks/br_team_map.csv`. Rows without a match keep approximate WAR and `war_source=approx`. See `docs/war_sources.md`.
