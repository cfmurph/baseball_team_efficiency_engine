# `@bos/api-client`

Typed client for the `#106` `/v1` contract. Used by `apps/web` now; Expo later.

- `GET /v1/health` → `{ as_of, active_season, current_season_missing, season_window }`
- `GET /v1/cards?season=&rec=` → schema 1.0 cards
- `GET /v1/seasons` → `{ seasons_present, season_window: [Y-2, Y], active_season }` (2024–2026 when Y=2026)
- `GET /v1/players?season=` → published `player_season_metrics` directory (default window `[Y-2, Y]`)
- `GET /v1/players/{id}` → one player; `{id}` is the internal `player_id` PK

If `baseUrl` is empty the client returns fixture cards (`fantasy/stub_cards.jsonl`) and a health object that can set `current_season_missing`. It does not invent 2026 rows when the API is live. Prior-year cards still ship; live 2026 is #131, not a client gate.
