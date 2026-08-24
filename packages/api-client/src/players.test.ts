import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_PLAYER_SORT,
  MIN_IP,
  MIN_PA,
  defaultDirectorySeason,
  formatAvg,
  formatOps,
  hittingLine,
  playerQualifies,
  pitchingLine,
  seasonWindowYears,
} from "./players.ts";
import { stubHealth } from "./client.ts";

test("friendly labels never leak raw SDIO keys", () => {
  assert.equal(formatAvg(0.322), ".322");
  assert.equal(formatOps(1.159), "1.159");
  const hit = hittingLine({
    season: 2024,
    g: 158,
    pa: 704,
    ab: 559,
    r: 122,
    h: 180,
    hr: 58,
    rbi: 144,
    sb: 10,
    bb: 133,
    so: 171,
    avg: 0.322,
    obp: 0.458,
    slg: 0.701,
    ops: 1.159,
    war: 10.8,
  });
  assert.equal(hit, ".322 AVG · 1.159 OPS · 58 HR");
  assert.equal(hit.includes("BattingAverage"), false);
  const pitch = pitchingLine({
    season: 2024,
    g: 31,
    gs: 31,
    ip: 192,
    w: 18,
    l: 4,
    sv: 0,
    so: 228,
    bb: 35,
    era: 2.39,
    whip: 0.92,
    war: 6.4,
  });
  assert.equal(pitch, "2.39 ERA · 0.92 WHIP · 228 K");
  assert.equal(pitch.includes("EarnedRunAverage"), false);
});

test("silent qualifier hides 12-PA flukes and keeps 20 IP", () => {
  assert.equal(MIN_PA, 50);
  assert.equal(MIN_IP, 20);
  assert.equal(DEFAULT_PLAYER_SORT, "war");
  assert.equal(playerQualifies({ pa: 12, ip: null }), false);
  assert.equal(playerQualifies({ pa: 50, ip: null }), true);
  assert.equal(playerQualifies({ pa: 0, ip: 20 }), true);
});

test("directory season falls back when current year is missing", () => {
  const health = stubHealth({ current_season_missing: true });
  assert.deepEqual(seasonWindowYears(health), [2024, 2025, 2026]);
  assert.equal(defaultDirectorySeason(health, [2024, 2025, 2026]), 2025);
  assert.equal(defaultDirectorySeason(stubHealth(), [2024, 2025, 2026]), 2026);
});
