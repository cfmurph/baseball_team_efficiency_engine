from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests

# Sports Reference and some CDNs reject the default Python-requests UA.
DEFAULT_HEADERS = {
    "User-Agent": (
        "baseball-team-efficiency-engine/1.0 "
        "(+https://github.com/cfmurph/baseball_team_efficiency_engine)"
    ),
    "Accept": "*/*",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_file(url: str, output_path: str | Path, timeout: int = 180) -> Path:
    """Download a URL to disk (CSV, TXT, or any other file)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def download_csv(url: str, output_path: str | Path) -> Path:
    return download_file(url, output_path, timeout=60)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
