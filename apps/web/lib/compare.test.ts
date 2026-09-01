import assert from "node:assert/strict";
import test from "node:test";

import type { PlayerDetail, PlayerListItem } from "@bos/api-client";

import {
  COMPARE_MAX,
  EMPTY_CELL,
  appendCompareId,
  bestIndexes,
  buildComparePath,
  buildCompareRows,
  clampSeason,
  columnFromDetail,
  compareHrefForPlayer,
  filterSlotCandidates,
  parseCompareIds,
  parseCompareMode,
  parseCompareQuery,
  readStoredCompare,
  removeCompareId,
  slotIds,
  writeStoredCompare,
} from "./compare.ts";

function memoryStorage(): { store: Map<string, string>; api: { getItem(key: string): string | null; setItem(key: string, value: string): void; removeItem(key: string): void } } {
  const store = new Map<string, string>();
  return {
    store,
    api: {
      getItem(key: string) {
        return store.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        store.set(key, value);
      },
      removeItem(key: string) {
        store.delete(key);
      },
    },
  };
}

function hitterDetail(): PlayerDetail {
  return {
    player: { player_id: "hitter-1", name: "Ada Hitter", position: "OF", team: "NYY" },
    hitting: [{
      season: 2025,
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
      doubles: 36,
      triples: 1,
      woba: 0.458,
      war: 10.8,
      war_source: "bbref",
    }],
    pitching: [],
    fielding: [{
      season: 2025,
      pos: "RF",
      g: 150,
      gs: 148,
      inn: 1305,
      po: 361,
      a: 8,
      e: 4,
      dp: 1,
      pb: null,
      fpct: 0.989,
    }],
    recent_games: { hitting: [], pitching: [] },
    card: null,
    source: "api",
  };
}

function pitcherDetail(): PlayerDetail {
  return {
    player: { player_id: "pitcher-1", name: "Pat Pitcher", position: "SP", team: "PHI" },
    hitting: [],
    pitching: [{
      season: 2025,
      g: 31,
      gs: 31,
      ip: 192,
      w: 18,
      l: 4,
      sv: 0,
      so: 228,
      bb: 35,
      er: 51,
      era: 2.39,
      whip: 0.92,
      fip: 2.81,
      war: 6.4,
      war_source: "bbref",
    }],
    fielding: [],
    recent_games: { hitting: [], pitching: [] },
    card: null,
    source: "api",
  };
}

test("parses shareable compare URL, caps at four, and defaults to players", () => {
  assert.deepEqual(parseCompareIds("a,b,a,c,d,e"), ["a", "b", "c", "d"]);
  assert.deepEqual(parseCompareIds(["x", "y"]), ["x", "y"]);
  assert.equal(parseCompareMode("teams"), "teams");
  assert.equal(parseCompareMode("players"), "players");
  assert.equal(parseCompareMode(""), "players");
  const query = parseCompareQuery(
    { mode: "teams", season: "2025", ids: "one,two,three,four,five" },
    2024,
    [2024, 2025, 2026],
  );
  assert.equal(query.mode, "players");
  assert.equal(query.season, 2025);
  assert.deepEqual(query.ids, ["one", "two", "three", "four"]);
  assert.equal(query.ids.length, COMPARE_MAX);
  assert.equal(clampSeason(2026, [2024, 2025], 2026), 2025);
  assert.equal(clampSeason(2025, [2024, 2025, 2026], 2026), 2025);
});

test("buildComparePath round-trips ids and season without minting ids", () => {
  const path = buildComparePath({ season: 2025, ids: ["hitter-1", "pitcher-1"] });
  assert.equal(path, "/compare?mode=players&season=2025&ids=hitter-1,pitcher-1");
  const parsed = parseCompareQuery(
    { mode: "players", season: "2025", ids: "hitter-1,pitcher-1" },
    2024,
    [2024, 2025],
  );
  assert.deepEqual(parsed.ids, ["hitter-1", "pitcher-1"]);
  assert.equal(parsed.season, 2025);
  assert.equal(buildComparePath({ season: 2025, ids: [] }), "/compare?mode=players&season=2025");
});

test("append blocks a fifth player and skips duplicates", () => {
  const four = ["a", "b", "c", "d"];
  assert.deepEqual(appendCompareId(four, "e"), four);
  assert.deepEqual(appendCompareId(["a", "b"], "a"), ["a", "b"]);
  assert.deepEqual(appendCompareId(["a"], "b"), ["a", "b"]);
  assert.deepEqual(removeCompareId(["a", "b"], "a"), ["b"]);
  assert.deepEqual(slotIds(["a", "b"]), ["a", "b", null, null]);
});

test("best-in-row highlights high WAR and inverted ERA/WHIP", () => {
  assert.deepEqual(bestIndexes([10.8, 6.4, null], "higher"), [0]);
  assert.deepEqual(bestIndexes([3.1, 2.39, 2.39], "lower"), [1, 2]);
  assert.deepEqual(bestIndexes([null, null], "higher"), []);
  assert.deepEqual(bestIndexes([0, 2], "higher"), [1]);
});

test("mixed sides keep both blocks and dash the opposite cells", () => {
  const columns = [
    columnFromDetail("hitter-1", hitterDetail(), 2025),
    columnFromDetail("pitcher-1", pitcherDetail(), 2025),
  ];
  const rows = buildCompareRows(columns);
  assert.ok(rows.some((row) => row.block === "hitting"));
  assert.ok(rows.some((row) => row.block === "pitching"));
  const hr = rows.find((row) => row.key === "hitting-hr");
  assert.equal(hr?.display[0], "58");
  assert.equal(hr?.display[1], EMPTY_CELL);
  assert.deepEqual(hr?.best, [0]);
  const era = rows.find((row) => row.key === "pitching-era");
  assert.equal(era?.display[0], EMPTY_CELL);
  assert.equal(era?.display[1], "2.39");
  assert.deepEqual(era?.best, [1]);
  assert.equal(era?.display.includes("0"), false);
  const missingYear = buildCompareRows([
    columnFromDetail("hitter-1", hitterDetail(), 2026),
    columnFromDetail("pitcher-1", pitcherDetail(), 2026),
  ]);
  assert.deepEqual(missingYear, []);
});

test("compareHrefForPlayer appends onto stored ids", () => {
  const { api } = memoryStorage();
  writeStoredCompare({ season: 2025, ids: ["hitter-1"] }, api);
  assert.equal(
    compareHrefForPlayer("pitcher-1", 2024, api),
    "/compare?mode=players&season=2025&ids=hitter-1,pitcher-1",
  );
  assert.equal(compareHrefForPlayer("third", 2025, api).includes("third"), true);
  assert.equal(compareHrefForPlayer("fourth", 2025, api).includes("fourth"), true);
  const stored = readStoredCompare(api);
  assert.deepEqual(stored?.ids, ["hitter-1", "pitcher-1", "third", "fourth"]);
  assert.equal(compareHrefForPlayer("extra-5", 2025, api).includes("extra-5"), false);
});

test("slot search matches name team pos and ranks by WAR", () => {
  const pool: PlayerListItem[] = [
    { player_id: "low", name: "Low War", position: "OF", team: "BOS", season: 2025, side: "hitting", pa: 200, ip: null, war: 1.1, edge: null, line: "", fpct: null, fielding_line: "" },
    { player_id: "tiny", name: "Twelve PA", position: "OF", team: "SEA", season: 2025, side: "hitting", pa: 12, ip: null, war: 8, edge: null, line: "", fpct: null, fielding_line: "" },
    { player_id: "high", name: "High War", position: "SP", team: "NYY", season: 2025, side: "pitching", pa: null, ip: 80, war: 4.2, edge: null, line: "", fpct: null, fielding_line: "" },
    { player_id: "taken", name: "Already In", position: "1B", team: "CIN", season: 2025, side: "hitting", pa: 300, ip: null, war: 5, edge: null, line: "", fpct: null, fielding_line: "" },
  ];
  const yankees = filterSlotCandidates(pool, "nyy", ["taken"]);
  assert.deepEqual(yankees.map((row) => row.player_id), ["high"]);
  const ranked = filterSlotCandidates(pool, "", ["taken"]);
  assert.deepEqual(ranked.map((row) => row.player_id), ["high", "low"]);
  assert.equal(ranked.some((row) => row.player_id === "tiny"), false);
});
