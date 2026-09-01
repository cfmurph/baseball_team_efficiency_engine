import assert from "node:assert/strict";
import test from "node:test";

import type { PlayerDetail } from "@bos/api-client";

import {
  EMPTY_FIELDING_COPY,
  fieldingForSeason,
  hasFieldingLine,
  hittingCells,
  hittingGameColumns,
  pitchingCells,
} from "./playerPage.ts";

function judgeDetail(): PlayerDetail {
  return {
    player: { player_id: "judgeaa01", name: "Aaron Judge", position: "OF", team: "NYY" },
    hitting: [{
      season: 2026,
      g: 120,
      pa: 500,
      ab: 400,
      r: 85,
      h: 140,
      doubles: 22,
      triples: 1,
      hr: 40,
      rbi: 100,
      sb: 8,
      bb: 90,
      so: 130,
      avg: 0.35,
      obp: 0.46,
      slg: 0.71,
      ops: 1.17,
      woba: 0.42,
      war: 6.1,
      war_source: "real",
    }],
    pitching: [],
    fielding: [{
      season: 2026,
      pos: "RF",
      g: 112,
      gs: 110,
      inn: 980,
      po: 248,
      a: 7,
      e: 3,
      dp: 2,
      pb: null,
      fpct: 0.988,
    }],
    recent_games: {
      hitting: [{
        date: "2026-08-22",
        opponent: "BOS",
        season: 2026,
        ab: 4,
        r: 1,
        h: 2,
        doubles: 1,
        triples: null,
        hr: 1,
        rbi: 3,
        sb: 0,
        bb: 1,
        so: 1,
      }],
      pitching: [],
    },
    card: null,
    source: "api",
  };
}

function dhDetail(): PlayerDetail {
  return {
    ...judgeDetail(),
    player: { player_id: "solerjo01", name: "Jorge Soler", position: "DH", team: "LAA" },
    fielding: [],
  };
}

test("player page tables expose batting and fielding from fixture-shaped data", () => {
  const detail = judgeDetail();
  const hitting = hittingCells(detail.hitting[0]);
  const labels = hitting.map((cell) => cell.label);
  assert.ok(labels.includes("G"));
  assert.ok(labels.includes("PA"));
  assert.ok(labels.includes("2B"));
  assert.ok(labels.includes("AVG"));
  assert.ok(labels.includes("OPS"));
  assert.ok(labels.includes("wOBA"));
  assert.ok(labels.includes("WAR"));
  const fielding = fieldingForSeason(detail, 2026);
  assert.equal(hasFieldingLine(fielding), true);
  assert.equal(fielding[0]?.pos, "RF");
  assert.equal(fielding[0]?.po, 248);
  const gameHeaders = hittingGameColumns(detail.recent_games.hitting);
  assert.ok(gameHeaders.includes("2B"));
  assert.equal(gameHeaders.includes("3B"), false);
});

test("fielding section stays honest when the season has no defensive line", () => {
  const fielding = fieldingForSeason(dhDetail(), 2026);
  assert.equal(hasFieldingLine(fielding), false);
  assert.equal(EMPTY_FIELDING_COPY, "No fielding line for this season");
});

test("pitching table includes counting and rates when present", () => {
  const cells = pitchingCells({
    season: 2024,
    g: 27,
    gs: 27,
    ip: 150.1,
    w: 12,
    l: 8,
    sv: 0,
    so: 145,
    bb: 42,
    er: 43,
    era: 2.57,
    whip: 1.05,
    fip: null,
    war: 1.8,
    war_source: "real",
  });
  const labels = cells.map((cell) => cell.label);
  assert.ok(labels.includes("IP"));
  assert.ok(labels.includes("W-L"));
  assert.ok(labels.includes("ERA"));
  assert.ok(labels.includes("WHIP"));
  assert.equal(labels.includes("FIP"), false);
});
