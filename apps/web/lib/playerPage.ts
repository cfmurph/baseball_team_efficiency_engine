import {
  formatAvg,
  formatCount,
  formatEra,
  formatFpct,
  formatIp,
  formatOps,
  formatWar,
  formatWhip,
  formatWl,
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

export function hittingCells(row: HittingSeason): StatCell[] {
  return [
    { key: "g", label: "G", value: formatCount(row.g) },
    { key: "pa", label: "PA", value: formatCount(row.pa) },
    { key: "ab", label: "AB", value: formatCount(row.ab) },
    { key: "r", label: "R", value: formatCount(row.r) },
    { key: "h", label: "H", value: formatCount(row.h) },
    { key: "doubles", label: "2B", value: formatCount(row.doubles) },
    { key: "triples", label: "3B", value: formatCount(row.triples) },
    { key: "hr", label: "HR", value: formatCount(row.hr) },
    { key: "rbi", label: "RBI", value: formatCount(row.rbi) },
    { key: "sb", label: "SB", value: formatCount(row.sb) },
    { key: "bb", label: "BB", value: formatCount(row.bb) },
    { key: "so", label: "SO", value: formatCount(row.so) },
    { key: "avg", label: "AVG", value: formatAvg(row.avg) },
    { key: "obp", label: "OBP", value: formatAvg(row.obp) },
    { key: "slg", label: "SLG", value: formatAvg(row.slg) },
    { key: "ops", label: "OPS", value: formatOps(row.ops) },
    { key: "woba", label: "wOBA", value: formatAvg(row.woba) },
    { key: "war", label: "WAR", value: formatWar(row.war) },
  ].filter((cell) => present(cell.value));
}

export function pitchingCells(row: PitchingSeason): StatCell[] {
  return [
    { key: "g", label: "G", value: formatCount(row.g) },
    { key: "gs", label: "GS", value: formatCount(row.gs) },
    { key: "ip", label: "IP", value: formatIp(row.ip) },
    { key: "wl", label: "W-L", value: formatWl(row) },
    { key: "sv", label: "SV", value: formatCount(row.sv) },
    { key: "so", label: "SO", value: formatCount(row.so) },
    { key: "bb", label: "BB", value: formatCount(row.bb) },
    { key: "er", label: "ER", value: formatCount(row.er) },
    { key: "era", label: "ERA", value: formatEra(row.era) },
    { key: "whip", label: "WHIP", value: formatWhip(row.whip) },
    { key: "fip", label: "FIP", value: formatEra(row.fip) },
    { key: "war", label: "WAR", value: formatWar(row.war) },
  ].filter((cell) => present(cell.value));
}

export function fieldingCells(row: FieldingSeason): StatCell[] {
  return [
    { key: "pos", label: "POS", value: row.pos || "—" },
    { key: "g", label: "G", value: formatCount(row.g) },
    { key: "gs", label: "GS", value: formatCount(row.gs) },
    { key: "inn", label: "INN", value: formatIp(row.inn) },
    { key: "po", label: "PO", value: formatCount(row.po) },
    { key: "a", label: "A", value: formatCount(row.a) },
    { key: "e", label: "E", value: formatCount(row.e) },
    { key: "dp", label: "DP", value: formatCount(row.dp) },
    { key: "fpct", label: "FPCT", value: formatFpct(row.fpct) },
    { key: "pb", label: "PB", value: formatCount(row.pb) },
  ];
}

export function fieldingForSeason(detail: PlayerDetail | null, season: number): FieldingSeason[] {
  if (!detail) {
    return [];
  }
  return detail.fielding.filter((row) => row.season === season);
}

export function hasFieldingLine(rows: FieldingSeason[]): boolean {
  return rows.some((row) => {
    const counts = [row.g, row.gs, row.inn, row.po, row.a, row.e, row.dp, row.pb, row.fpct];
    return Boolean(row.pos) || counts.some((value) => value !== null);
  });
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
