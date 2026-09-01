import {
  formatAvg,
  formatCount,
  formatEra,
  formatFpct,
  formatIp,
  formatOps,
  formatPct,
  formatPerNine,
  formatRatio,
  formatWar,
  formatWhip,
  withFieldingIdentities,
  withHittingIdentities,
  withPitchingIdentities,
  type FieldingSeason,
  type HittingGame,
  type HittingSeason,
  type PitchingGame,
  type PitchingSeason,
  type PlayerDetail,
} from "@bos/api-client";

export const EMPTY_FIELDING_COPY = "No fielding line for this season";

export type StatCell = {
  key: string;
  label: string;
  value: string;
};

function present(value: string): boolean {
  return value !== "—";
}

function keepPresent(cells: StatCell[]): StatCell[] {
  return cells.filter((cell) => present(cell.value));
}

export function hittingCells(row: HittingSeason): StatCell[] {
  return keepPresent(hittingStandardCells(row));
}

export function hittingStandardCells(row: HittingSeason): StatCell[] {
  const line = withHittingIdentities(row);
  return keepPresent([
    { key: "g", label: "G", value: formatCount(line.g) },
    { key: "pa", label: "PA", value: formatCount(line.pa) },
    { key: "ab", label: "AB", value: formatCount(line.ab) },
    { key: "r", label: "R", value: formatCount(line.r) },
    { key: "h", label: "H", value: formatCount(line.h) },
    { key: "singles", label: "1B", value: formatCount(line.singles) },
    { key: "doubles", label: "2B", value: formatCount(line.doubles) },
    { key: "triples", label: "3B", value: formatCount(line.triples) },
    { key: "hr", label: "HR", value: formatCount(line.hr) },
    { key: "xbh", label: "XBH", value: formatCount(line.xbh) },
    { key: "tb", label: "TB", value: formatCount(line.tb) },
    { key: "rbi", label: "RBI", value: formatCount(line.rbi) },
    { key: "sb", label: "SB", value: formatCount(line.sb) },
    { key: "cs", label: "CS", value: formatCount(line.cs) },
    { key: "sb_pct", label: "SB%", value: formatPct(line.sb_pct) },
    { key: "bb", label: "BB", value: formatCount(line.bb) },
    { key: "ibb", label: "IBB", value: formatCount(line.ibb) },
    { key: "so", label: "SO", value: formatCount(line.so) },
    { key: "hbp", label: "HBP", value: formatCount(line.hbp) },
    { key: "sh", label: "SH", value: formatCount(line.sh) },
    { key: "sf", label: "SF", value: formatCount(line.sf) },
    { key: "gidp", label: "GIDP", value: formatCount(line.gidp) },
    { key: "roe", label: "ROE", value: formatCount(line.roe) },
    { key: "lob", label: "LOB", value: formatCount(line.lob) },
    { key: "gsh", label: "GSH", value: formatCount(line.gsh) },
    { key: "go_ao", label: "GO/AO", value: formatRatio(line.go_ao) },
    { key: "avg", label: "AVG", value: formatAvg(line.avg) },
    { key: "obp", label: "OBP", value: formatAvg(line.obp) },
    { key: "slg", label: "SLG", value: formatAvg(line.slg) },
    { key: "ops", label: "OPS", value: formatOps(line.ops) },
  ]);
}

export function hittingAdvancedCells(row: HittingSeason): StatCell[] {
  const line = withHittingIdentities(row);
  return keepPresent([
    { key: "woba", label: "wOBA", value: formatAvg(line.woba) },
    { key: "war", label: "WAR", value: formatWar(line.war) },
    { key: "iso", label: "ISO", value: formatAvg(line.iso) },
    { key: "babip", label: "BABIP", value: formatAvg(line.babip) },
    { key: "k_pct", label: "K%", value: formatPct(line.k_pct) },
    { key: "bb_pct", label: "BB%", value: formatPct(line.bb_pct) },
  ]);
}

export function pitchingCells(row: PitchingSeason): StatCell[] {
  return keepPresent(pitchingStandardCells(row));
}

export function pitchingStandardCells(row: PitchingSeason): StatCell[] {
  const line = withPitchingIdentities(row);
  return keepPresent([
    { key: "g", label: "App", value: formatCount(line.g) },
    { key: "gs", label: "GS", value: formatCount(line.gs) },
    { key: "cg", label: "CG", value: formatCount(line.cg) },
    { key: "sho", label: "SHO", value: formatCount(line.sho) },
    { key: "qs", label: "QS", value: formatCount(line.qs) },
    { key: "gf", label: "GF", value: formatCount(line.gf) },
    { key: "w", label: "W", value: formatCount(line.w) },
    { key: "l", label: "L", value: formatCount(line.l) },
    { key: "wpct", label: "WPCT", value: formatAvg(line.wpct) },
    { key: "sv", label: "SV", value: formatCount(line.sv) },
    { key: "svo", label: "SVO", value: formatCount(line.svo) },
    { key: "sv_pct", label: "SV%", value: formatPct(line.sv_pct) },
    { key: "bs", label: "BS", value: formatCount(line.bs) },
    { key: "hld", label: "HLD", value: formatCount(line.hld) },
    { key: "ip", label: "IP", value: formatIp(line.ip) },
    { key: "h", label: "H", value: formatCount(line.h) },
    { key: "r", label: "R", value: formatCount(line.r) },
    { key: "er", label: "ER", value: formatCount(line.er) },
    { key: "uer", label: "UER", value: formatCount(line.uer) },
    { key: "hr", label: "HR", value: formatCount(line.hr) },
    { key: "bb", label: "BB", value: formatCount(line.bb) },
    { key: "so", label: "SO", value: formatCount(line.so) },
    { key: "era", label: "ERA", value: formatEra(line.era) },
    { key: "whip", label: "WHIP", value: formatWhip(line.whip) },
    { key: "wp", label: "WP", value: formatCount(line.wp) },
    { key: "bk", label: "BK", value: formatCount(line.bk) },
    { key: "bf", label: "BF", value: formatCount(line.bf) },
    { key: "np", label: "NP", value: formatCount(line.np) },
    { key: "pk", label: "PK", value: formatCount(line.pk) },
    { key: "ir", label: "IR", value: formatCount(line.ir) },
    { key: "go_ao", label: "GO/AO", value: formatRatio(line.go_ao) },
  ]);
}

export function pitchingAdvancedCells(row: PitchingSeason): StatCell[] {
  const line = withPitchingIdentities(row);
  return keepPresent([
    { key: "fip", label: "FIP", value: formatEra(line.fip) },
    { key: "war", label: "WAR", value: formatWar(line.war) },
    { key: "k9", label: "K/9", value: formatPerNine(line.k9) },
    { key: "bb9", label: "BB/9", value: formatPerNine(line.bb9) },
    { key: "h9", label: "H/9", value: formatPerNine(line.h9) },
    { key: "hr9", label: "HR/9", value: formatPerNine(line.hr9) },
    { key: "k_bb", label: "K/BB", value: formatRatio(line.k_bb) },
    { key: "k_pct", label: "K%", value: formatPct(line.k_pct) },
    { key: "bb_pct", label: "BB%", value: formatPct(line.bb_pct) },
    { key: "i_gs", label: "I/GS", value: formatIp(line.i_gs) },
  ]);
}

export function fieldingCells(row: FieldingSeason): StatCell[] {
  return keepPresent(fieldingStandardCells(row));
}

export function fieldingStandardCells(row: FieldingSeason): StatCell[] {
  const line = withFieldingIdentities(row);
  const cells = [
    { key: "pos", label: "POS", value: line.pos || "—" },
    { key: "g", label: "G", value: formatCount(line.g) },
    { key: "gs", label: "GS", value: formatCount(line.gs) },
    { key: "inn", label: "INN", value: formatIp(line.inn) },
    { key: "po", label: "PO", value: formatCount(line.po) },
    { key: "a", label: "A", value: formatCount(line.a) },
    { key: "e", label: "E", value: formatCount(line.e) },
    { key: "dp", label: "DP", value: formatCount(line.dp) },
    { key: "fpct", label: "FPCT", value: formatFpct(line.fpct) },
    { key: "pb", label: "PB", value: formatCount(line.pb) },
    { key: "tc", label: "TC", value: formatCount(line.tc) },
    { key: "tp", label: "TP", value: formatCount(line.tp) },
    { key: "ofa", label: "OFA", value: formatCount(line.ofa) },
    { key: "cs_pct", label: "CS%", value: formatPct(line.cs_pct) },
  ];
  return cells.filter((cell) => cell.key === "pos" || cell.key === "g" || cell.key === "gs" || present(cell.value));
}

export function fieldingAdvancedCells(row: FieldingSeason): StatCell[] {
  const line = withFieldingIdentities(row);
  return keepPresent([{ key: "rf", label: "RF", value: formatPerNine(line.rf) }]);
}

export function fieldingForSeason(detail: PlayerDetail | null, season: number): FieldingSeason[] {
  if (!detail) {
    return [];
  }
  return detail.fielding.filter((row) => row.season === season);
}

export function hasFieldingLine(rows: FieldingSeason[]): boolean {
  return rows.some((row) => {
    const filled = withFieldingIdentities(row);
    const counts = [
      filled.g, filled.gs, filled.inn, filled.po, filled.a, filled.e, filled.dp,
      filled.pb, filled.fpct, filled.ofa, filled.cs, filled.tp, filled.tc, filled.rf,
    ];
    return Boolean(filled.pos) || counts.some((value) => value !== null && value !== undefined);
  });
}

export function fieldingHeaderCells(rows: FieldingSeason[]): StatCell[] {
  if (!rows.length) {
    return [];
  }
  const presentKeys = new Set<string>();
  for (const row of rows) {
    for (const cell of fieldingStandardCells(row)) {
      presentKeys.add(cell.key);
    }
  }
  return fieldingStandardCells(rows[0]).filter((cell) => presentKeys.has(cell.key));
}

export function hittingGameColumns(games: HittingGame[]): string[] {
  const extra = {
    doubles: games.some((game) => game.doubles !== null),
    triples: games.some((game) => game.triples !== null),
    sb: games.some((game) => game.sb !== null),
  };
  return [
    "Date",
    "Opp",
    "AB",
    "R",
    "H",
    ...(extra.doubles ? ["2B"] : []),
    ...(extra.triples ? ["3B"] : []),
    "HR",
    "RBI",
    ...(extra.sb ? ["SB"] : []),
    "BB",
    "K",
  ];
}

export function pitchingGameColumns(games: PitchingGame[]): string[] {
  const extra = {
    hr: games.some((game) => game.hr !== null),
    gs: games.some((game) => game.gs !== null),
    decision: games.some((game) => Boolean(game.decision)),
  };
  return [
    "Date",
    "Opp",
    ...(extra.gs ? ["GS"] : []),
    "IP",
    "H",
    ...(extra.hr ? ["HR"] : []),
    "ER",
    "BB",
    "K",
    ...(extra.decision ? ["Dec"] : []),
  ];
}
