import assert from "node:assert/strict";
import test from "node:test";

import { shouldShowSeasonBanner } from "@bos/card-schema";

test("re-exports banner rule used by the page loader", () => {
  const health = {
    as_of: "2026-08-23",
    active_season: 2026,
    current_season_missing: false,
    season_window: [2024, 2025, 2026],
    source: "local" as const,
  };
  assert.equal(shouldShowSeasonBanner(health, [2024, 2025, 2026]), false);
  assert.equal(
    shouldShowSeasonBanner({ ...health, current_season_missing: true }, [2024], {
      liveFeed: false,
    }),
    false,
  );
  assert.equal(
    shouldShowSeasonBanner({ ...health, seasons_present: [2024, 2025] }, []),
    true,
  );
});
