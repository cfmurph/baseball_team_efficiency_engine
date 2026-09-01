import assert from "node:assert/strict";
import test from "node:test";

import { defaultDirectorySeason } from "@bos/api-client";
import { shouldShowSeasonBanner } from "@bos/card-schema";

import { loadCompareData } from "./load.ts";

test("re-exports banner rule used by the page loader", () => {
  assert.equal(
    shouldShowSeasonBanner(
      {
        as_of: "2026-08-23",
        active_season: 2026,
        current_season_missing: false,
        season_window: { start: 2024, end: 2026 },
      },
      [2024, 2025, 2026],
    ),
    false,
  );
});

test("player directory defaults to the latest published year", () => {
  const health = {
    as_of: "2026-08-23",
    active_season: 2026,
    current_season_missing: true,
    season_window: { start: 2024, end: 2026 },
  };
  assert.equal(defaultDirectorySeason(health, [2024, 2025, 2026]), 2025);
  assert.equal(
    shouldShowSeasonBanner(health, [2024, 2025, 2026]),
    true,
  );
});

test("player page does not treat a published prior year as missing", () => {
  const health = {
    as_of: "2026-08-23",
    active_season: 2026,
    current_season_missing: true,
    season_window: { start: 2024, end: 2026 },
  };
  assert.equal(shouldShowSeasonBanner(health, [2024, 2025]), true);
});

test("compare loader keeps missing players empty in stub mode", async () => {
  const data = await loadCompareData({
    mode: "players",
    season: "2025",
    ids: "hitter-1,pitcher-1",
  });
  assert.equal(data.query.mode, "players");
  assert.deepEqual(data.query.ids, ["hitter-1", "pitcher-1"]);
  assert.equal(data.details.length, 2);
  assert.equal(data.details[0], null);
  assert.equal(data.details[1], null);
  assert.ok(data.seasons.includes(data.query.season));
});
