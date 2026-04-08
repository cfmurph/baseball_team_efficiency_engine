"""
Sportradar MLB data extractor.

Pulls seasonal player statistics, transactions, and injuries from the
Sportradar MLB v8 API and loads them into the DuckDB warehouse.

Authentication
--------------
Set SPORTRADAR_API_KEY in your environment before running:

    export SPORTRADAR_API_KEY=your_key_here
    python3 -m pipeline.extract.pull_sportradar

Or add it to Cursor Cloud Agent Secrets so it's injected automatically.

Usage
-----
    # Pull stats for 2024 regular season
    python3 -m pipeline.extract.pull_sportradar --year 2024

    # Pull multiple years
    python3 -m pipeline.extract.pull_sportradar --year 2022 --year 2023 --year 2024

    # Pull transactions for a date range
    python3 -m pipeline.extract.pull_sportradar --transactions-since 2024-01-01

    # Pull current injuries
    python3 -m pipeline.extract.pull_sportradar --injuries

    # Dry-run: fetch but do not write to warehouse
    python3 -m pipeline.extract.pull_sportradar --year 2024 --dry-run

    # Use a local response cache to avoid re-hitting the API during dev
    python3 -m pipeline.extract.pull_sportradar --year 2024 --cache-dir data/sr_cache
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import typer
from typing_extensions import Annotated

from src.baseball_analytics.config import load_settings
from src.baseball_analytics.io import ensure_dir
from src.baseball_analytics.schema import WAREHOUSE_DDL
from src.baseball_analytics.sportradar import SportradarClient, SportradarError

log = logging.getLogger(__name__)
app = typer.Typer(add_completion=False, help="Pull Sportradar MLB data into the warehouse.")


# ---------------------------------------------------------------------------
# Parsing helpers — flatten nested Sportradar JSON into DataFrames
# ---------------------------------------------------------------------------

def _parse_team_map(hierarchy: dict) -> pd.DataFrame:
    """Flatten league hierarchy into a team-crosswalk DataFrame."""
    rows = []
    for league in hierarchy.get("leagues", []):
        for division in league.get("divisions", []):
            for team in division.get("teams", []):
                rows.append({
                    "sr_team_id": team["id"],
                    "sr_abbr": team.get("abbr", ""),
                    "sr_market": team.get("market", ""),
                    "sr_name": team.get("name", ""),
                    "lahman_team_id": None,   # filled from static crosswalk
                    "lahman_franch_id": None,
                })
    return pd.DataFrame(rows)


def _parse_player_season(team_stats: dict, sr_team_id: str, year: int) -> pd.DataFrame:
    """
    Flatten Sportradar seasonal_stats response into a player-season DataFrame.
    Handles both hitters and pitchers; missing fields become NaN.

    The v8 seasonal stats response has players at the top level of the response
    dict (not nested under a 'team' key).
    """
    rows = []
    # Players are at the top level in v8
    players = team_stats.get("players", [])

    for p in players:
        base = {
            "sr_player_id": p.get("id"),
            "sr_team_id": sr_team_id,
            "season_year": year,
            "full_name": p.get("full_name") or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip(),
            "position": p.get("position"),
            "primary_position": p.get("primary_position"),
            "jersey_number": p.get("jersey_number"),
        }

        # ---- Hitting ----
        hit = (
            p.get("statistics", {})
            .get("hitting", {})
            .get("overall", {})
        )
        hit_onbase = hit.get("onbase", {})
        hit_outs = hit.get("outs", {})
        hit_steal = hit.get("steal", {})

        base.update({
            "pa":       hit.get("ap"),
            "ab":       hit.get("ab"),
            "hits":     hit_onbase.get("h"),
            "doubles":  hit_onbase.get("d"),
            "triples":  hit_onbase.get("t"),
            "hr":       hit_onbase.get("hr"),
            "rbi":      hit.get("rbi"),
            "bb":       hit_onbase.get("bb"),
            "ibb":      hit_onbase.get("ibb"),
            "hbp":      hit_onbase.get("hbp"),
            "sb":       hit_steal.get("stolen"),
            "avg":      hit.get("avg"),
            "obp":      hit.get("obp"),
            "slg":      hit.get("slg"),
            "ops":      hit.get("ops"),
            "woba":     hit.get("woba"),
            "wraa":     hit.get("wraa"),
            "wrc":      hit.get("wrc"),
            "wrc_plus": hit.get("wrc_plus"),
            "war":      hit.get("war"),
            "bwar":     hit.get("bwar"),
            "brwar":    hit.get("brwar"),
            "fwar":     hit.get("fwar"),
        })

        # ---- Pitching ----
        pit = (
            p.get("statistics", {})
            .get("pitching", {})
            .get("overall", {})
        )
        pit_onbase = pit.get("onbase", {})

        base.update({
            "ip":       pit.get("ip_2"),
            "era":      pit.get("era"),
            "era_minus": pit.get("era_minus"),
            "fip":      pit.get("fip"),
            "whip":     pit.get("whip"),
            "k9":       pit.get("k9"),
            "bb9":      pit.get("bb9"),
            "hr9":      pit_onbase.get("hr9"),
            "kbb":      pit.get("kbb"),
            "p_war":    pit.get("war"),
        })

        rows.append(base)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "sr_player_id" not in df.columns:
        return pd.DataFrame()
    return df[df["sr_player_id"].notna()].reset_index(drop=True)


def _parse_transactions(tx_data: dict) -> pd.DataFrame:
    """Flatten a daily transactions response into a DataFrame."""
    rows = []
    for player in tx_data.get("players", []):
        player_id = player.get("id")
        player_name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        for tx in player.get("transactions", []):
            rows.append({
                "transaction_id":   tx.get("id"),
                "effective_date":   tx.get("effective_date"),
                "last_modified":    tx.get("last_modified"),
                "transaction_type": tx.get("transaction_type"),
                "transaction_code": tx.get("transaction_code"),
                "description":      tx.get("desc"),
                "sr_player_id":     player_id,
                "player_name":      player_name,
                "from_team_abbr":   tx.get("from_team", {}).get("abbr"),
                "to_team_abbr":     tx.get("to_team", {}).get("abbr"),
                "from_sr_team_id":  tx.get("from_team", {}).get("id"),
                "to_sr_team_id":    tx.get("to_team", {}).get("id"),
            })
    return pd.DataFrame(rows)


def _parse_injuries(injury_data: dict, fetched_at: str) -> pd.DataFrame:
    """Flatten the current injuries response into a DataFrame."""
    rows = []
    for team in injury_data.get("teams", []):
        team_id = team.get("id")
        team_abbr = team.get("abbr")
        for player in team.get("players", []):
            injury = player.get("injury", {})
            rows.append({
                "sr_player_id":  player.get("id"),
                "player_name":   player.get("full_name") or f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
                "sr_team_id":    team_id,
                "team_abbr":     team_abbr,
                "injury_desc":   injury.get("desc"),
                "injury_status": injury.get("status"),
                "start_date":    injury.get("start_date"),
                "end_date":      injury.get("end_date"),
                "fetched_at":    fetched_at,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Warehouse loader
# ---------------------------------------------------------------------------

def _load_df(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> int:
    """Insert DataFrame into table using column-aligned insert. Returns row count."""
    if df.empty:
        log.warning("Empty DataFrame for %s — skipping", table)
        return 0
    db_cols = [r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()]
    common = [c for c in df.columns if c in db_cols]
    if not common:
        log.error("No overlapping columns for %s", table)
        return 0
    view = f"_sr_{table}"
    con.register(view, df[common])
    col_list = ", ".join(common)
    con.execute(f"INSERT OR REPLACE INTO {table} ({col_list}) SELECT {col_list} FROM {view}")
    con.unregister(view)
    return len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    config_path: str = typer.Option("config/settings.yaml", help="Path to settings YAML"),
    year: Annotated[
        Optional[list[int]], typer.Option("--year", help="Season year(s) to pull stats for")
    ] = None,
    season_type: str = typer.Option("REG", help="Season type: REG, PRE, PST"),
    transactions_since: Annotated[
        Optional[str], typer.Option(help="Pull transactions from this date (YYYY-MM-DD) to today")
    ] = None,
    injuries: bool = typer.Option(False, "--injuries/--no-injuries", help="Pull current injury report"),
    access_level: str = typer.Option("trial", help="Sportradar access level: trial or production"),
    cache_dir: Annotated[
        Optional[str], typer.Option(help="Directory to cache raw API responses")
    ] = None,
    dry_run: bool = typer.Option(False, "--dry-run/--no-dry-run", help="Fetch but do not write to warehouse"),
    api_key: Annotated[
        Optional[str], typer.Option("--api-key", help="Sportradar API key (overrides SPORTRADAR_API_KEY env var)")
    ] = None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings(config_path)

    client = SportradarClient(
        api_key=api_key,
        access_level=access_level,
        cache_dir=cache_dir,
    )

    warehouse_path = Path(settings["warehouse_path"])
    ensure_dir(warehouse_path.parent)
    con = duckdb.connect(str(warehouse_path))
    con.execute(WAREHOUSE_DDL)

    # ---- Team crosswalk ----
    static_crosswalk = Path("data/crosswalks/sportradar_team_map.csv")
    if static_crosswalk.exists():
        cw_df = pd.read_csv(static_crosswalk)
        if not dry_run:
            _load_df(con, "dim_sportradar_team_map", cw_df)
            log.info("Loaded %d rows into dim_sportradar_team_map", len(cw_df))
    else:
        log.warning("Static crosswalk not found at %s — loading from API hierarchy", static_crosswalk)
        hierarchy = client.league_hierarchy()
        cw_df = _parse_team_map(hierarchy)
        if not dry_run:
            _load_df(con, "dim_sportradar_team_map", cw_df)

    # Build a quick lookup: sr_team_id → lahman_team_id
    team_lookup: dict[str, str] = {}
    if not cw_df.empty and "sr_team_id" in cw_df.columns:
        team_lookup = cw_df.set_index("sr_team_id")["lahman_team_id"].dropna().to_dict()

    # ---- Seasonal player stats ----
    if year:
        for yr in year:
            log.info("Pulling %s %d seasonal stats for %d teams", season_type, yr, len(cw_df))
            all_players: list[pd.DataFrame] = []

            for _, row in cw_df.iterrows():
                sr_team_id = row["sr_team_id"]
                try:
                    data = client.seasonal_stats(sr_team_id, yr, season_type)
                    team_df = _parse_player_season(data, sr_team_id, yr)
                    if not team_df.empty:
                        all_players.append(team_df)
                    log.info("  %s %s (%d players)", row.get("sr_abbr", "?"), yr, len(team_df))
                except SportradarError as exc:
                    log.warning("  Skipping %s %d: %s", row.get("sr_abbr", sr_team_id), yr, exc)

            if all_players:
                season_df = pd.concat(all_players, ignore_index=True)
                season_df = season_df.drop_duplicates(["sr_player_id", "season_year", "sr_team_id"])
                if not dry_run:
                    n = _load_df(con, "fact_sr_player_season", season_df)
                    log.info("Loaded %d player-season rows for %d", n, yr)
                else:
                    log.info("[dry-run] Would load %d rows for %d", len(season_df), yr)

    # ---- Transactions ----
    if transactions_since:
        start = date.fromisoformat(transactions_since)
        end = date.today()
        current = start
        tx_frames: list[pd.DataFrame] = []
        log.info("Pulling transactions %s → %s", start, end)

        while current <= end:
            try:
                data = client.transactions(current.year, current.month, current.day)
                df_tx = _parse_transactions(data)
                if not df_tx.empty:
                    tx_frames.append(df_tx)
            except SportradarError as exc:
                log.warning("  Transactions %s: %s", current, exc)
            current += timedelta(days=1)

        if tx_frames:
            all_tx = pd.concat(tx_frames, ignore_index=True).drop_duplicates("transaction_id")
            if not dry_run:
                n = _load_df(con, "fact_sr_transactions", all_tx)
                log.info("Loaded %d transaction rows", n)
            else:
                log.info("[dry-run] Would load %d transaction rows", len(all_tx))

    # ---- Injuries ----
    if injuries:
        import datetime
        fetched_at = datetime.datetime.utcnow().isoformat()
        try:
            data = client.injuries()
            injury_df = _parse_injuries(data, fetched_at)
            if not dry_run:
                con.execute("DELETE FROM fact_sr_injuries")
                n = _load_df(con, "fact_sr_injuries", injury_df)
                log.info("Loaded %d injury rows", n)
            else:
                log.info("[dry-run] Would load %d injury rows", len(injury_df))
        except SportradarError as exc:
            log.error("Injuries fetch failed: %s", exc)

    con.close()
    typer.echo("Sportradar pull complete.")


if __name__ == "__main__":
    app()
