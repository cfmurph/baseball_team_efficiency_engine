from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def download_csv(url: str, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return output_path


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
