"""``python3 -m services.api`` — read-only API over published ``current/``."""
from __future__ import annotations

import uvicorn

from services.api.app import bind_host, bind_port


def main() -> None:
    uvicorn.run(
        "services.api.app:app",
        host=bind_host(),
        port=bind_port(),
        factory=False,
    )


if __name__ == "__main__":
    main()
