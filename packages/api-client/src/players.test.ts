import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_PLAYER_SORT,
  MIN_IP,
  MIN_PA,
  defaultDirectorySeason,
  deriveRf,
  deriveSbPct,
  deriveSingles,
  deriveTb,
  deriveTc,
  deriveXbh,
  formatAvg,
  formatOps,
  hittingCountingLine,
  hittingLine,
  hittingRatesLine,
  isApproxWar,
  parsePlayerDetail,
  parsePlayersList,
  playerQualifies,
  pitchingLine,
  seasonWindowYears,
  selectedYearMissing,
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
    doubles: 36,
    triples: 1,
    woba: 0.458,
    war: 10.8,
    war_source: "bbref",
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
    er: 51,
    era: 2.39,
    whip: 0.92,
    fip: 2.81,
    war: 6.4,
    war_source: "bbref",
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

test("counting and rates stay friendly and never say vs repl", () => {
  const row = {
    season: 2026,
    g: 118,
    pa: 528,
    ab: 420,
    r: 88,
    h: 132,
    hr: 41,
    rbi: 98,
    sb: 7,
    bb: 95,
    so: 128,
    avg: 0.314,
    obp: 0.445,
    slg: 0.688,
    ops: 1.133,
    doubles: 22,
    triples: 1,
    woba: 0.445,
    war: 7.1,
    war_source: "bbref",
  };
  const counting = hittingCountingLine(row);
  const rates = hittingRatesLine(row);
  assert.match(counting, /118 G/);
  assert.match(counting, /41 HR/);
  assert.match(rates, /\.314 AVG/);
  assert.equal(counting.includes("vs repl"), false);
  assert.equal(rates.includes("woba"), false);
  assert.equal(isApproxWar("approx"), true);
  assert.equal(isApproxWar("real"), false);
});

test("parses #152 PlayerRecord and keeps an empty game log honest", () => {
  const detail = parsePlayerDetail({
    as_of: "2026-08-23",
    active_season: 2026,
    current_season_missing: false,
    season_window: [2024, 2025, 2026],
    source: "local",
    player: {
      player_id: "judgeaa01",
      name: "Aaron Judge",
      position: "OF",
      team: "NYY",
      seasons: [
        {
          season: 2026,
          team: "NYY",
          player_type: "batter",
          war_source: "real",
          war: 6.1,
          games: 120,
          pa: 500,
          hits: 140,
          hr: 40,
          rbi: 100,
          sb: 8,
          avg: 0.35,
        },
      ],
    },
  });
  assert.equal(detail?.player.player_id, "judgeaa01");
  assert.equal(detail?.hitting[0]?.hr, 40);
  assert.deepEqual(detail?.recent_games.hitting, []);
  assert.deepEqual(detail?.fielding, []);
  const list = parsePlayersList(
    {
      players: [
        {
          player_id: "judgeaa01",
          name: "Aaron Judge",
          position: "OF",
          team: "NYY",
          seasons: [{ season: 2026, pa: 500, war: 6.1, avg: 0.35, hr: 40, ops: 1.1 }],
        },
      ],
    },
    2026,
  );
  assert.equal(list[0]?.player_id, "judgeaa01");
  assert.equal(list[0]?.war, 6.1);
  assert.equal(list[0]?.hitting?.hr, 40);
  assert.equal(list[0]?.hitting?.avg, 0.35);
  assert.equal(list[0]?.hitting?.ops, 1.1);
  assert.equal(list[0]?.hitting?.pa, 500);
});

test("parses fielding lines when present and omits them when absent", () => {
  const withFielding = parsePlayerDetail({
    player: {
      player_id: "judgeaa01",
      name: "Aaron Judge",
      position: "OF",
      team: "NYY",
      seasons: [
        {
          season: 2026,
          player_type: "batter",
          pa: 500,
          hits: 140,
          ab: 400,
          hr: 40,
          fielding: [
            { pos: "RF", g: 112, po: 248, a: 7, e: 3, dp: 2, fpct: 0.988 },
          ],
        },
      ],
    },
  });
  assert.equal(withFielding?.fielding.length, 1);
  assert.equal(withFielding?.fielding[0]?.pos, "RF");
  assert.equal(withFielding?.fielding[0]?.po, 248);
  assert.equal(withFielding?.fielding[0]?.fpct, 0.988);

  const emptyFielding = parsePlayerDetail({
    player: {
      player_id: "solerjo01",
      name: "Jorge Soler",
      position: "DH",
      team: "LAA",
      seasons: [{ season: 2026, player_type: "batter", pa: 210, hits: 40, ab: 180 }],
    },
  });
  assert.deepEqual(emptyFielding?.fielding, []);
});

test("derived identities need every input and stay off otherwise", () => {
  assert.equal(deriveSingles(140, 22, 1, 40), 77);
  assert.equal(deriveXbh(22, 1, 40), 63);
  assert.equal(deriveTb(140, 22, 1, 40), 284);
  assert.equal(deriveTc(248, 7, 3), 258);
  assert.ok(Math.abs((deriveSbPct(8, 3) ?? 0) - 8 / 11) < 1e-9);
  assert.ok(Math.abs((deriveRf(248, 7, 980) ?? 0) - (9 * 255) / 980) < 1e-9);
  assert.equal(deriveSingles(140, 22, null, 40), null);
  assert.equal(deriveXbh(22, null, 40), null);
  assert.equal(deriveTb(140, null, 1, 40), null);
  assert.equal(deriveSbPct(8, null), null);
  assert.equal(deriveTc(248, 7, null), null);
  assert.equal(deriveRf(248, 7, null), null);
});

test("directory list items keep published season numbers, not only rec strings", () => {
  const list = parsePlayersList(
    {
      players: [
        {
          player_id: "judgeaa01",
          name: "Aaron Judge",
          position: "OF",
          team: "NYY",
          seasons: [{
            season: 2026,
            player_type: "batter",
            games: 120,
            pa: 500,
            ab: 400,
            runs: 85,
            hits: 140,
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
            fielding: [{ pos: "RF", g: 112, po: 248, a: 7, e: 3, dp: 2, fpct: 0.988 }],
          }],
        },
        {
          player_id: "suarera02",
          name: "Ranger Suárez",
          position: "SP",
          team: "PHI",
          seasons: [{
            season: 2026,
            player_type: "pitcher",
            pa: 0,
            games: 27,
            gs: 27,
            ip: 150.1,
            w: 12,
            l: 8,
            so: 145,
            pitching_so: 145,
            pitching_bb: 42,
            er: 43,
            era: 2.57,
            whip: 1.05,
            pitching_hits: 120,
            cg: 2,
            qs: 18,
            bf: 610,
            war: 1.8,
            war_source: "real",
          }],
        },
      ],
    },
    2026,
  );
  const judge = list.find((row) => row.player_id === "judgeaa01");
  const suarez = list.find((row) => row.player_id === "suarera02");
  assert.equal(judge?.hitting?.hr, 40);
  assert.equal(judge?.hitting?.doubles, 22);
  assert.equal(judge?.hitting?.singles, 77);
  assert.equal(judge?.hitting?.ops, 1.17);
  assert.equal(judge?.fielding[0]?.pos, "RF");
  assert.equal(judge?.fielding[0]?.fpct, 0.988);
  assert.equal(judge?.pitching, null);
  assert.match(judge?.line || "", /AVG/);
  assert.notEqual(judge?.line, undefined);
  assert.ok(judge?.hitting);
  assert.equal(suarez?.side, "pitching");
  assert.equal(suarez?.hitting, null);
  assert.equal(suarez?.pitching?.era, 2.57);
  assert.equal(suarez?.pitching?.ip, 150.1);
  assert.equal(suarez?.pitching?.cg, 2);
  assert.equal(suarez?.pitching?.qs, 18);
  assert.equal(suarez?.pitching?.bf, 610);
});

test("banner only when the selected year is the missing current season", () => {
  const health = stubHealth({ current_season_missing: true });
  assert.equal(selectedYearMissing(health, 2026, false), true);
  assert.equal(selectedYearMissing(health, 2025, false), false);
  assert.equal(selectedYearMissing(health, 2026, true), false);
});
