import assert from "node:assert/strict";
import test from "node:test";

import { createApiClient, defaultSeasonYears, stubCardsFeed, stubHealth } from "./client.ts";

test("stub feed is the same four schema 1.0 cards", () => {
  const cards = stubCardsFeed();
  assert.equal(cards.length, 4);
  assert.deepEqual(
    cards.map((card) => card.recommendation_type).sort(),
    ["pickup", "sit", "start", "stream"],
  );
  const sit = cards.find((card) => card.recommendation_type === "sit");
  assert.equal(sit?.edge?.is_approx, true);
  assert.equal(sit?.edge?.war_source, "approx");
  assert.equal(sit?.season, 2026);
});

test("default season window is [Y-2, Y] for 2026", () => {
  assert.deepEqual(defaultSeasonYears(2026), [2024, 2025, 2026]);
});

test("unset API URL uses fixtures and can set current_season_missing", async () => {
  const client = createApiClient({ stubCurrentSeasonMissing: true });
  assert.equal(client.source, "stub");
  const health = await client.getHealth();
  assert.equal(health.current_season_missing, true);
  assert.equal(health.active_season, 2026);
  assert.deepEqual(health.season_window, { start: 2024, end: 2026 });
  const seasons = await client.getSeasons();
  assert.deepEqual(seasons.seasons, [2024, 2025, 2026]);
  const cards = await client.getCards();
  assert.equal(cards.source, "stub");
  assert.equal(cards.cards.length, 4);
  const sitOnly = await client.getCards({ rec: "sit" });
  assert.equal(sitOnly.cards.length, 1);
  assert.equal(sitOnly.cards[0]?.player?.name, "Jorge Soler");
});

test("live client hits /v1 without inventing rows", async () => {
  const calls: string[] = [];
  const client = createApiClient({
    baseUrl: "https://api.example.test/",
    fetch: async (input) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith("/v1/health")) {
        return new Response(
          JSON.stringify({
            as_of: "2025-09-01",
            active_season: 2026,
            current_season_missing: true,
            season_window: [2024, 2026],
          }),
        );
      }
      if (url.includes("/v1/cards")) {
        return new Response(JSON.stringify({ cards: [] }));
      }
      if (url.endsWith("/v1/seasons")) {
        return new Response(
          JSON.stringify({
            as_of: "2025-09-01",
            active_season: 2026,
            season_window: [2024, 2025, 2026],
            seasons_present: [2024, 2025],
            current_season_missing: true,
          }),
        );
      }
      return new Response("missing", { status: 404 });
    },
  });
  assert.equal(client.source, "api");
  const health = await client.getHealth();
  assert.equal(health.current_season_missing, true);
  const cards = await client.getCards({ season: 2026, rec: "start" });
  assert.equal(cards.source, "api");
  assert.deepEqual(cards.cards, []);
  const seasons = await client.getSeasons();
  assert.deepEqual(seasons.seasons, [2024, 2025]);
  assert.ok(calls.some((url) => url.includes("/v1/cards?season=2026&rec=start")));
});

test("stubHealth override stays explicit", () => {
  assert.equal(stubHealth().current_season_missing, false);
  assert.equal(
    stubHealth({ current_season_missing: true }).current_season_missing,
    true,
  );
});
