import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveRf,
  deriveSbPct,
  deriveSingles,
  deriveTb,
  deriveTc,
  deriveXbh,
  type PlayerDetail,
} from "@bos/api-client";

import {
  EMPTY_FIELDING_COPY,
  fieldingForSeason,
  hasFieldingLine,
  hittingAdvancedCells,
  hittingCells,
  hittingGameColumns,
  pitchingAdvancedCells,
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
      cs: 3,
      hbp: 8,
      sf: 5,
      gidp: 10,
      ibb: 12,
      sh: 0,
      lob: 80,
      roe: 2,
      gsh: 2,
      go: 90,
      ao: 80,
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
  for (const label of ["G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "XBH", "TB", "RBI", "SB", "CS", "SB%", "BB", "IBB", "SO", "HBP", "SF", "GIDP", "AVG", "OBP", "SLG", "OPS"]) {
    assert.ok(labels.includes(label), `missing ${label}`);
  }
  assert.equal(labels.includes("WO"), false);
  const advanced = hittingAdvancedCells(detail.hitting[0]).map((cell) => cell.label);
  assert.ok(advanced.includes("wOBA"));
  assert.ok(advanced.includes("WAR"));
  assert.ok(advanced.includes("ISO"));
  assert.ok(advanced.includes("BABIP"));
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

test("pitching table includes standard extras and hides missing advanced", () => {
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
    h: 120,
    hr: 12,
    r: 48,
    cg: 2,
    sho: 1,
    hld: 0,
    bs: 0,
    qs: 18,
    gf: 0,
    bk: 1,
    wp: 4,
    bf: 610,
    np: 2300,
    go: 180,
    ao: 120,
  });
  const labels = cells.map((cell) => cell.label);
  assert.ok(labels.includes("App"));
  assert.ok(labels.includes("IP"));
  assert.ok(labels.includes("W"));
  assert.ok(labels.includes("L"));
  assert.ok(labels.includes("WPCT"));
  assert.ok(labels.includes("H"));
  assert.ok(labels.includes("CG"));
  assert.ok(labels.includes("QS"));
  assert.ok(labels.includes("BF"));
  assert.ok(labels.includes("ERA"));
  assert.ok(labels.includes("WHIP"));
  assert.ok(labels.includes("GO/AO"));
  const advanced = pitchingAdvancedCells({
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
    h: 120,
    hr: 12,
    bf: 610,
  }).map((cell) => cell.label);
  assert.equal(advanced.includes("FIP"), false);
  assert.ok(advanced.includes("WAR"));
  assert.ok(advanced.includes("K/9"));
});

test("derived columns stay off when inputs are missing", () => {
  assert.equal(deriveSingles(140, 22, null, 40), null);
  assert.equal(deriveXbh(22, null, 40), null);
  assert.equal(deriveTb(140, null, 1, 40), null);
  assert.equal(deriveSbPct(8, null), null);
  assert.equal(deriveTc(248, 7, null), null);
  assert.equal(deriveRf(248, 7, null), null);
  const labels = hittingCells({
    season: 2026,
    g: 120,
    pa: 500,
    ab: 400,
    r: 85,
    h: 140,
    doubles: null,
    triples: null,
    hr: 40,
    rbi: 100,
    sb: 8,
    bb: 90,
    so: 130,
    avg: 0.35,
    obp: 0.46,
    slg: 0.71,
    ops: 1.17,
    woba: null,
    war: 6.1,
    war_source: "real",
  }).map((cell) => cell.label);
  assert.equal(labels.includes("1B"), false);
  assert.equal(labels.includes("XBH"), false);
  assert.equal(labels.includes("TB"), false);
  assert.equal(labels.includes("SB%"), false);
});
