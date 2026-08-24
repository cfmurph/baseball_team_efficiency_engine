# Testing

The suite is a three-layer pyramid. Markers are registered in `pytest.ini`.
Every test has exactly one of `unit`, `integration`, or `e2e`.

## Layers

| Layer | Marker | What belongs here |
|---|---|---|
| Unit | `unit` | Pure functions, mappers, schema, `war_source`, `share.stat_line`, config, metrics helpers. No network. No heavy I/O. |
| Integration | `integration` | Warehouse / metrics / storage / fantasy emitter / MLB Stats API / SportsDataIO ingest / thin read API with fixtures or `file://` backends. Nightly `PIPELINE_STEPS` contract (`pull_war` after `pull_sources`, `pull_mlb_stats` after `pull_war`, `pull_sportsdataio` after Stats API). |
| E2E | `e2e` | Streamlit AppTest (all GM nav pages + BenchOrStart boot), golden rWAR spot checks, and fantasy `cards.jsonl` path under `current/` and `runs/{run_id}/`. |

No layer talks to live Baseball-Reference, MLB Stats API, SportsDataIO, or object storage. E2E uses committed fixtures and in-process AppTest only.

## Run locally

```bash
source .venv/bin/activate
pip install -r requirements.txt

python3 -m pytest -m unit -v
python3 -m pytest -m integration -v
python3 -m pytest -m e2e -v

# Full suite (same tests as the three CI jobs combined)
python3 -m pytest tests/ -v
```

`--strict-markers` is on by default (`pytest.ini`). Unknown markers fail collection.

## What CI enforces

PRs to `master` run `.github/workflows/ci.yml` as **four separate checks**:

| Check name | Command | Supersedes from `ci-smoke.yml` |
|---|---|---|
| **Unit tests** | `pytest -m unit` | BenchOrStart copy lock (`tests/test_web_copy_lock.py`) |
| **Integration tests** | `pytest -m integration` | Nightly pipeline contract (`tests/test_run_nightly.py`) + SportsDataIO ingest (`tests/test_sportsdataio.py`) + read API (`tests/test_api.py`, including `/v1/players`) |
| **E2E tests** | `pytest -m e2e` | AppTest (`tests/test_dashboard_apptest.py`) + golden WAR (`tests/test_golden_war.py`) |
| **BenchOrStart Next.js** | `npm install && npm test && npm run build` | Next.js job from the old `ci-smoke.yml` |

The old single job **Dashboard + pipeline + golden WAR** is a thin alias in `ci.yml` that depends on the three pyramid jobs (the master ruleset still requires that exact name). Its coverage is split:

- AppTest all sidebar pages → **E2E tests**
- Golden WAR fixtures (Judge 2022, Trout 2012, deGrom 2018, Ohtani 2023) → **E2E tests**
- Nightly `PIPELINE_STEPS` / fail-fast / publish-after-success → **Integration tests**

The unit job also fails if any test is missing a layer marker.

Jobs are independent so GitHub shows separate required-style checks. Wall clock stays well under 15 minutes (the Python suite is a few seconds on a warm checkout).

## Adding a test

1. Put it next to the existing module tests when possible.
2. Tag it with `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.e2e` (or inherit `pytestmark` from the file).
3. Do not add a second layer marker. Overlap would run the test in two CI jobs.
4. Do not hit the network. Inject fetchers / runners, or use `tests/fixtures/` and `file://`.
