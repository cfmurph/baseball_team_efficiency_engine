import assert from "node:assert/strict";
import test from "node:test";

import {
  createApiClient,
  defaultSeasonYears,
  parseSeasonWindow,
  stubCardsFeed,
  stubHealth,
  stubSeasons,
  probeLocalV1,
} from "./client.ts";

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
  assert.deepEqual(parseSeasonWindow([2024, 2025, 2026]), [2024, 2025, 2026]);
});

test("unset API URL uses fixtures and can set current_season_missing", async () => {
  const client = createApiClient({ stubCurrentSeasonMissing: true });
  assert.equal(client.source, "stub");
  const health = await client.getHealth();
  assert.equal(health.current_season_missing, true);
  assert.equal(health.active_season, 2026);
  assert.deepEqual(health.season_window, [2024, 2025, 2026]);
  assert.equal(health.source, "local");
  assert.deepEqual(health.seasons_present, [2024, 2025]);
  const seasons = await client.getSeasons();
  assert.equal(seasons.as_of, "2026-08-23");
  assert.deepEqual(seasons.season_window, [2024, 2025, 2026]);
  assert.deepEqual(seasons.seasons_present, [2024, 2025]);
  assert.equal(seasons.current_season_missing, true);
  const cards = await client.getCards();
  assert.equal(cards.source, "stub");
  assert.equal(cards.schema_version, "1.0");
  assert.equal(cards.cards.length, 4);
  const sitOnly = await client.getCards({ rec: "sit" });
  assert.equal(sitOnly.cards.length, 1);
  assert.equal(sitOnly.rec, "sit");
  assert.equal(sitOnly.cards[0]?.player?.name, "Jorge Soler");
  const players = await client.getPlayers({ season: 2026 });
  assert.deepEqual(players.players, []);
  assert.equal(players.current_season_missing, true);
  const player = await client.getPlayer("judgeaa01");
  assert.equal(player.player, null);
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
            season_window: [2024, 2025, 2026],
            source: "local",
            seasons_present: [2024, 2025],
            current_season_missing_reason: "sdio_unavailable",
          }),
        );
      }
      if (url.includes("/v1/cards")) {
        return new Response(
          JSON.stringify({
            schema_version: "1.0",
            as_of: "2025-09-01",
            season: 2026,
            rec: "start",
            current_season_missing: true,
            cards: [],
          }),
        );
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
      if (url.includes("/v1/players/judgeaa01")) {
        return new Response(
          JSON.stringify({
            as_of: "2025-09-01",
            active_season: 2026,
            current_season_missing: true,
            season_window: [2024, 2025, 2026],
            seasons_present: [2024, 2025],
            source: "local",
            player: { player_id: "judgeaa01", name: "Aaron Judge", seasons: [] },
          }),
        );
      }
      if (url.includes("/v1/players")) {
        return new Response(
          JSON.stringify({
            as_of: "2025-09-01",
            active_season: 2026,
            current_season_missing: true,
            season_window: [2024, 2025, 2026],
            seasons_present: [2024, 2025],
            source: "local",
            players: [],
          }),
        );
      }
      return new Response("missing", { status: 404 });
    },
  });
  assert.equal(client.source, "api");
  const health = await client.getHealth();
  assert.equal(health.current_season_missing, true);
  assert.deepEqual(health.season_window, [2024, 2025, 2026]);
  assert.deepEqual(health.seasons_present, [2024, 2025]);
  assert.equal(health.seasons_present?.includes(2026), false);
  assert.equal(health.source, "local");
  assert.equal(health.current_season_missing_reason, "sdio_unavailable");
  const cards = await client.getCards({ season: 2026, rec: "start" });
  assert.equal(cards.source, "api");
  assert.equal(cards.schema_version, "1.0");
  assert.equal(cards.as_of, "2025-09-01");
  assert.equal(cards.season, 2026);
  assert.equal(cards.rec, "start");
  assert.equal(cards.current_season_missing, true);
  assert.deepEqual(cards.cards, []);
  const seasons = await client.getSeasons();
  assert.equal(seasons.as_of, "2025-09-01");
  assert.deepEqual(seasons.season_window, [2024, 2025, 2026]);
  assert.deepEqual(seasons.seasons_present, [2024, 2025]);
  assert.equal(seasons.current_season_missing, true);
  const players = await client.getPlayers({ season: 2026 });
  assert.deepEqual(players.players, []);
  assert.equal(players.current_season_missing, true);
  const player = await client.getPlayer("judgeaa01", { season: 2026 });
  assert.equal(player.player?.player_id, "judgeaa01");
  assert.deepEqual(player.player?.seasons, []);
  assert.ok(calls.some((url) => url.includes("/v1/cards?season=2026&rec=start")));
  assert.ok(calls.some((url) => url.includes("/v1/players?season=2026")));
  assert.ok(calls.some((url) => url.includes("/v1/players/judgeaa01?season=2026")));
});

test("empty cards envelope is a miss, not a stub", async () => {
  const client = createApiClient({
    baseUrl: "https://api.example.test",
    fetch: async () =>
      new Response(
        JSON.stringify({
          schema_version: "1.0",
          as_of: "2025-09-01",
          current_season_missing: true,
          cards: [],
        }),
      ),
  });
  const cards = await client.getCards();
  assert.equal(cards.source, "api");
  assert.deepEqual(cards.cards, []);
  assert.equal(cards.current_season_missing, true);
});

test("prior-year API cards ship as-is; share.stat_line stays verbatim", async () => {
  const prior = {
    schema_version: "1.0",
    card_id: "prior-start-1",
    recommendation_type: "start",
    season: 2025,
    player: { name: "Prior Year" },
    share: { stat_line: "+2.1 edge · 80% conf" },
  };
  const client = createApiClient({
    baseUrl: "https://api.example.test",
    fetch: async (input) => {
      const url = String(input);
      if (url.endsWith("/v1/cards")) {
        return new Response(
          JSON.stringify({
            schema_version: "1.0",
            as_of: "2025-09-01",
            current_season_missing: true,
            cards: [prior],
          }),
        );
      }
      return new Response("missing", { status: 404 });
    },
  });
  const cards = await client.getCards();
  assert.equal(cards.source, "api");
  assert.equal(cards.cards.length, 1);
  assert.equal(cards.cards[0]?.season, 2025);
  assert.equal(cards.cards[0]?.share?.stat_line, "+2.1 edge · 80% conf");
  assert.equal(cards.cards.some((card) => Number(card.season) === 2026), false);
});

test("live local /v1 is not the stub feed when the process is up", async () => {
  let health: Response;
  try {
    health = await fetch("http://127.0.0.1:8000/v1/health", {
      headers: { accept: "application/json" },
    });
  } catch {
    return;
  }
  if (!health.ok) {
    return;
  }
  const client = createApiClient({ baseUrl: "http://127.0.0.1:8000" });
  const [seasons, cards] = await Promise.all([
    client.getSeasons(),
    client.getCards(),
  ]);
  assert.equal(client.source, "api");
  assert.equal(cards.source, "api");
  assert.equal(cards.schema_version, "1.0");
  assert.equal(cards.cards.some((card) => String(card.card_id || "").startsWith("stub-")), false);
  assert.ok(Array.isArray(seasons.season_window));
  assert.ok(Array.isArray(seasons.seasons_present));
});

test("probeLocalV1 prefers a live /v1 health and falls back when down", async () => {
  const up = await probeLocalV1(async (input) => {
    assert.equal(String(input), "http://127.0.0.1:8000/v1/health");
    return new Response(JSON.stringify({ as_of: "2026-08-23" }), { status: 200 });
  });
  assert.equal(up, "http://127.0.0.1:8000");
  const down = await probeLocalV1(async () => {
    throw new Error("ECONNREFUSED");
  });
  assert.equal(down, null);
});

test("stubHealth override stays explicit", () => {
  assert.equal(stubHealth().current_season_missing, false);
  assert.deepEqual(stubHealth().season_window, [2024, 2025, 2026]);
  assert.equal(stubHealth().source, "local");
  assert.equal(
    stubHealth({ current_season_missing: true }).current_season_missing,
    true,
  );
  assert.equal(stubSeasons(2026, { stubCurrentSeasonMissing: true }).current_season_missing, true);
});
