import assert from "node:assert/strict";
import test from "node:test";

import {
  parsePlayersList,
  type FieldingSeason,
  type HittingSeason,
  type PitchingSeason,
  type PlayerListItem,
} from "@bos/api-client";

import {
  directoryFieldingColumns,
  directoryHittingAdvancedColumns,
  directoryHittingColumns,
  directoryPitchingAdvancedColumns,
  directoryPitchingColumns,
  directoryRowsForSide,
  hittingStandardCells,
  pitchingStandardCells,
  statValue,
} from "./playerPage.ts";

function judgeHitting(): HittingSeason {
  return {
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
  };
}

function judgeFielding(): FieldingSeason {
  return {
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
  };
}

function suarezPitching(): PitchingSeason {
  return {
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
  };
}

function listItem(partial: Partial<PlayerListItem> & Pick<PlayerListItem, "player_id" | "name" | "side">): PlayerListItem {
  return {
    position: "",
    team: "",
    season: 2026,
    pa: null,
    ip: null,
    war: null,
    edge: null,
    line: "",
    fpct: null,
    fielding_line: "",
    hitting: null,
    pitching: null,
    fielding: [],
    ...partial,
  };
}

test("directory helpers render glossary standard batting columns from a fixture batter", () => {
  const hitting = judgeHitting();
  const labels = directoryHittingColumns([hitting]).map((cell) => cell.label);
  for (const label of ["G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "XBH", "TB", "RBI", "SB", "AVG", "OBP", "SLG", "OPS"]) {
    assert.ok(labels.includes(label), `missing ${label}`);
  }
  const cells = hittingStandardCells(hitting);
  assert.equal(statValue(cells, "avg"), ".350");
  assert.equal(statValue(cells, "ops"), "1.170");
  assert.equal(statValue(cells, "hr"), "40");
  assert.equal(statValue(cells, "pa"), "500");
  const fieldLabels = directoryFieldingColumns([judgeFielding()]).map((cell) => cell.label);
  assert.ok(fieldLabels.includes("POS"));
  assert.ok(fieldLabels.includes("FPCT"));
  assert.ok(fieldLabels.includes("TC"));
  const advanced = directoryHittingAdvancedColumns([hitting]).map((cell) => cell.label);
  assert.ok(advanced.includes("wOBA"));
  assert.ok(advanced.includes("WAR"));
  assert.equal(labels.includes("DRS"), false);
  assert.equal(labels.includes("UZR"), false);
  assert.equal(labels.includes("OAA"), false);
  assert.equal(labels.includes("OPS+"), false);
  assert.equal(labels.includes("wRC+"), false);
});

test("directory helpers render glossary standard pitching columns from a fixture pitcher", () => {
  const pitching = suarezPitching();
  const labels = directoryPitchingColumns([pitching]).map((cell) => cell.label);
  for (const label of ["App", "GS", "IP", "W", "L", "SO", "ERA", "WHIP"]) {
    assert.ok(labels.includes(label), `missing ${label}`);
  }
  const cells = pitchingStandardCells(pitching);
  assert.equal(statValue(cells, "era"), "2.57");
  assert.equal(statValue(cells, "ip"), "150.1");
  assert.equal(statValue(cells, "so"), "145");
  const advanced = directoryPitchingAdvancedColumns([pitching]).map((cell) => cell.label);
  assert.equal(advanced.includes("FIP"), false);
  assert.ok(advanced.includes("WAR"));
  assert.ok(advanced.includes("K/9"));
  assert.equal(labels.includes("ERA+"), false);
  assert.equal(labels.includes("Statcast"), false);
});

test("compact line strings are no longer the only stats shown", () => {
  const list = parsePlayersList(
    {
      players: [{
        player_id: "judgeaa01",
        name: "Aaron Judge",
        position: "OF",
        team: "NYY",
        seasons: [{
          season: 2026,
          player_type: "batter",
          pa: 500,
          ab: 400,
          hits: 140,
          doubles: 22,
          triples: 1,
          hr: 40,
          avg: 0.35,
          ops: 1.17,
          war: 6.1,
        }],
      }],
    },
    2026,
  );
  const row = list[0];
  assert.ok(row?.line.includes("AVG"));
  assert.ok(row?.hitting);
  assert.equal(row?.hitting?.hr, 40);
  assert.equal(row?.hitting?.doubles, 22);
  const labels = directoryHittingColumns(row.hitting ? [row.hitting] : []).map((cell) => cell.label);
  assert.ok(labels.includes("PA"));
  assert.ok(labels.includes("AB"));
  assert.ok(labels.includes("2B"));
  assert.ok(labels.includes("OPS"));
  assert.ok(labels.length > 3);
});

test("two-way players keep both sides and empty columns stay hidden", () => {
  const twoWay = listItem({
    player_id: "ohtansh01",
    name: "Shohei Ohtani",
    position: "DH",
    team: "LAD",
    side: "hitting",
    hitting: judgeHitting(),
    pitching: suarezPitching(),
  });
  assert.equal(directoryRowsForSide([twoWay], "hitting").length, 1);
  assert.equal(directoryRowsForSide([twoWay], "pitching").length, 1);
  const emptyIso: HittingSeason = {
    ...judgeHitting(),
    woba: null,
    doubles: null,
    triples: null,
    hr: 40,
    h: 140,
  };
  const labels = directoryHittingColumns([emptyIso]).map((cell) => cell.label);
  assert.equal(labels.includes("1B"), false);
  assert.equal(labels.includes("XBH"), false);
  assert.equal(labels.includes("TB"), false);
  const advanced = directoryHittingAdvancedColumns([emptyIso]).map((cell) => cell.label);
  assert.equal(advanced.includes("wOBA"), false);
});
