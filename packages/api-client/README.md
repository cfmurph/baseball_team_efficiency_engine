# `@bos/api-client`

Typed client for the merged master `/v1` contract (`services/api/openapi.yaml`, #144 / #106 / #152). Used by `apps/web` now; Expo later.

- `GET /v1/health` → `{ as_of, active_season, current_season_missing, season_window, source: remote|local|missing, seasons_present?, current_season_missing_reason? }`
- `GET /v1/seasons` → `{ as_of, active_season, season_window, seasons_present, current_season_missing }`
- `GET /v1/cards?season=&rec=` → `{ schema_version: "1.0", as_of, season?, rec?: start|sit|pickup|stream, current_season_missing, cards }`
- `GET /v1/players?season=` → published `player_season_metrics` directory (default window `[Y-2, Y]`)
- `GET /v1/players/{id}` → one player; `{id}` is the internal `player_id` PK

`season_window` is the product default `[Y-2, Y]` (2024–2026 when Y=2026). `seasons_present` is years that actually exist — 2026 can be in the window and absent from `seasons_present`. Empty `cards` / `players` is a miss, not a stub. Never invent 2026 rows. `share.stat_line` is verbatim from the API.

If `baseUrl` is empty the client prefers `http://127.0.0.1:8000` when that process answers `/v1/health`. Fixture cards (`fantasy/stub_cards.jsonl`) only if the API process is down. Prior-year cards still ship; live 2026 is #131, not a client gate.
