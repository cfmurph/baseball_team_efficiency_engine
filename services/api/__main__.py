"""``python3 -m services.api`` — read-only API over published ``current/``."""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.environ.get("API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    raw_port = os.environ.get("API_PORT", "8000").strip() or "8000"
    try:
        port = int(raw_port)
    except ValueError:
        port = 8000
    uvicorn.run("services.api.app:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
