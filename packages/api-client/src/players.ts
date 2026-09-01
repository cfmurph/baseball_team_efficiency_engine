import {
  DEFAULT_ACTIVE_SEASON,
  cardReason,
  recommendationLabel,
  type FantasyCard,
  type Health,
} from "@bos/card-schema";

export const MIN_PA = 50;
export const MIN_IP = 20;
export const DEFAULT_PLAYER_SORT = "war";
export const RECENT_GAME_LIMIT = 6;

export type PlayerSide = "hitting" | "pitching";
export type PlayerSort = "war" | "edge";

export type PlayerIdentity = {
  player_id: string;
  name: string;
  position: string;
  team: string;
};

export type HittingSeason = {
  season: number;
  g: number | null;
  pa: number | null;
  ab: number | null;
  r: number | null;
  h: number | null;
  doubles: number | null;
  triples: number | null;
  hr: number | null;
  rbi: number | null;
  sb: number | null;
  bb: number | null;
  so: number | null;
  avg: number | null;
  obp: number | null;
  slg: number | null;
  ops: number | null;
  woba: number | null;
  war: number | null;
  war_source: string;
  singles?: number | null;
  xbh?: number | null;
  tb?: number | null;
  cs?: number | null;
  sb_pct?: number | null;
  hbp?: number | null;
  sh?: number | null;
  sf?: number | null;
  gidp?: number | null;
  ibb?: number | null;
  lob?: number | null;
  roe?: number | null;
  gsh?: number | null;
  go?: number | null;
  ao?: number | null;
  go_ao?: number | null;
  iso?: number | null;
  babip?: number | null;
  k_pct?: number | null;
  bb_pct?: number | null;
};

export type PitchingSeason = {
  season: number;
  g: number | null;
  gs: number | null;
  ip: number | null;
  w: number | null;
  l: number | null;
  sv: number | null;
  so: number | null;
  bb: number | null;
  er: number | null;
  era: number | null;
  whip: number | null;
  fip: number | null;
  war: number | null;
  war_source: string;
  h?: number | null;
  hr?: number | null;
  r?: number | null;
  uer?: number | null;
  cg?: number | null;
  sho?: number | null;
  hld?: number | null;
  bs?: number | null;
  svo?: number | null;
  sv_pct?: number | null;
  qs?: number | null;
  gf?: number | null;
  bk?: number | null;
  wp?: number | null;
  np?: number | null;
  pk?: number | null;
  ir?: number | null;
  bf?: number | null;
  go?: number | null;
  ao?: number | null;
  go_ao?: number | null;
  wpct?: number | null;
  k9?: number | null;
  bb9?: number | null;
  h9?: number | null;
  hr9?: number | null;
  k_bb?: number | null;
  k_pct?: number | null;
  bb_pct?: number | null;
  i_gs?: number | null;
};

export type FieldingSeason = {
  season: number;
  pos: string;
  g: number | null;
  gs: number | null;
  inn: number | null;
  po: number | null;
  a: number | null;
  e: number | null;
  dp: number | null;
  pb: number | null;
  fpct: number | null;
  ofa?: number | null;
  cs?: number | null;
  sb?: number | null;
  tp?: number | null;
  tc?: number | null;
  rf?: number | null;
  cs_pct?: number | null;
};

/** #152 published season row on GET /v1/players and /v1/players/{id}. */
export type PlayerSeason = {
  season: number;
  team?: string | null;
  team_name?: string | null;
  position?: string | null;
  player_type?: string | null;
  stat_source?: string | null;
  war_source?: string | null;
  war?: number | null;
  games?: number | null;
  pa?: number | null;
  ab?: number | null;
  hits?: number | null;
  hr?: number | null;
  bb?: number | null;
  so?: number | null;
  rbi?: number | null;
  sb?: number | null;
  ip?: number | null;
  pitching_so?: number | null;
  pitching_bb?: number | null;
  avg?: number | null;
  obp?: number | null;
  slg?: number | null;
  ops?: number | null;
  woba?: number | null;
  era?: number | null;
  whip?: number | null;
  fip?: number | null;
  runs?: number | null;
  doubles?: number | null;
  triples?: number | null;
  gs?: number | null;
  w?: number | null;
  l?: number | null;
  sv?: number | null;
  er?: number | null;
  putouts?: number | null;
  assists?: number | null;
  errors?: number | null;
  double_plays?: number | null;
  passed_balls?: number | null;
  fielding_g?: number | null;
  fielding_gs?: number | null;
  fielding_inn?: number | null;
  fielding_pos?: string | null;
  fpct?: number | null;
  fielding?: FieldingSeason[] | Array<Record<string, unknown>>;
  cs?: number | null;
  hbp?: number | null;
  sh?: number | null;
  sf?: number | null;
  gidp?: number | null;
  ibb?: number | null;
  lob?: number | null;
  roe?: number | null;
  gsh?: number | null;
  singles?: number | null;
  tb?: number | null;
  xbh?: number | null;
  go?: number | null;
  ao?: number | null;
  ofa?: number | null;
  fielding_cs?: number | null;
  fielding_sb?: number | null;
  tp?: number | null;
  tc?: number | null;
  rf?: number | null;
  pitching_hits?: number | null;
  pitching_hr?: number | null;
  pitching_r?: number | null;
  cg?: number | null;
  sho?: number | null;
  hld?: number | null;
  bs?: number | null;
  svo?: number | null;
  qs?: number | null;
  gf?: number | null;
  bk?: number | null;
  wp?: number | null;
  np?: number | null;
  pk?: number | null;
  ir?: number | null;
  uer?: number | null;
  bf?: number | null;
  pitching_go?: number | null;
  pitching_ao?: number | null;
  iso?: number | null;
  babip?: number | null;
  sb_pct?: number | null;
  go_ao?: number | null;
  k_pct?: number | null;
  bb_pct?: number | null;
  wpct?: number | null;
  sv_pct?: number | null;
  pitching_go_ao?: number | null;
  k9?: number | null;
  bb9?: number | null;
  h9?: number | null;
  hr9?: number | null;
  k_bb?: number | null;
  pitching_k_pct?: number | null;
  pitching_bb_pct?: number | null;
  i_gs?: number | null;
  cs_pct?: number | null;
};

export type PlayerRecord = {
  player_id: string;
  name?: string | null;
  position?: string | null;
  team?: string | null;
  seasons: PlayerSeason[];
};

export type HittingGame = {
  date: string;
  opponent: string;
  season: number;
  ab: number | null;
  r: number | null;
  h: number | null;
  doubles: number | null;
  triples: number | null;
  hr: number | null;
  rbi: number | null;
  sb: number | null;
  bb: number | null;
  so: number | null;
};

export type PitchingGame = {
  date: string;
  opponent: string;
  season: number;
  ip: number | null;
  h: number | null;
  er: number | null;
  hr: number | null;
  bb: number | null;
  so: number | null;
  gs: number | null;
  decision: string | null;
};

export type PlayerCardChip = {
  recommendation_type: string;
  label: string;
  reason: string;
};

export type PlayerDetail = {
  player: PlayerIdentity;
  hitting: HittingSeason[];
  pitching: PitchingSeason[];
  fielding: FieldingSeason[];
  recent_games: {
    hitting: HittingGame[];
    pitching: PitchingGame[];
  };
  card: PlayerCardChip | null;
  source: "api" | "stub";
};

export type PlayerListItem = {
  player_id: string;
  name: string;
  position: string;
  team: string;
  season: number;
  side: PlayerSide;
  pa: number | null;
  ip: number | null;
  war: number | null;
  edge: number | null;
  line: string;
  fpct: number | null;
  fielding_line: string;
};

export type PlayersQuery = {
  season?: number;
  sort?: PlayerSort | string;
  min_pa?: number;
  min_ip?: number;
  q?: string;
  position?: string;
  team?: string;
};

export type PlayersResponse = {
  players: PlayerListItem[];
  season: number;
  sort: string;
  source: "api" | "stub";
};

export type StubPlayerRecord = {
  player: PlayerIdentity;
  hitting?: HittingSeason[];
  pitching?: PitchingSeason[];
  fielding?: FieldingSeason[];
  recent_games?: {
    hitting?: HittingGame[];
    pitching?: PitchingGame[];
  };
};

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function asList(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function num(value: unknown): number | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function text(value: unknown): string {
  if (value === undefined || value === null) {
    return "";
  }
  return String(value).trim();
}

export function formatAvg(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  const body = value.toFixed(3);
  return value < 1 && value >= 0 ? body.replace(/^0/, "") : body;
}

export function formatOps(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(3);
}

export function formatEra(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(2);
}

export function formatWhip(value: number | null | undefined): string {
  return formatEra(value);
}

export function formatCount(value: number | null | undefined, digits = 0): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return digits === 0 ? String(Math.round(value)) : value.toFixed(digits);
}

export function formatIp(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(1);
}

export function formatWar(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(1);
}

export function formatWl(row: Pick<PitchingSeason, "w" | "l">): string {
  if (row.w === null && row.l === null) {
    return "—";
  }
  return `${formatCount(row.w)}–${formatCount(row.l)}`;
}

export function formatFpct(value: number | null | undefined): string {
  return formatAvg(value);
}

export function formatPct(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return `${(value * 100).toFixed(1)}%`;
}

export function formatPerNine(value: number | null | undefined): string {
  if (value === undefined || value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(2);
}

export function formatRatio(value: number | null | undefined): string {
  return formatPerNine(value);
}

function allPresent(...values: Array<number | null | undefined>): boolean {
  return values.every((value) => value !== undefined && value !== null && Number.isFinite(value));
}

export function deriveSingles(
  hits: number | null | undefined,
  doubles: number | null | undefined,
  triples: number | null | undefined,
  hr: number | null | undefined,
): number | null {
  if (!allPresent(hits, doubles, triples, hr)) {
    return null;
  }
  return Math.max(0, (hits as number) - (doubles as number) - (triples as number) - (hr as number));
}

export function deriveXbh(
  doubles: number | null | undefined,
  triples: number | null | undefined,
  hr: number | null | undefined,
): number | null {
  if (!allPresent(doubles, triples, hr)) {
    return null;
  }
  return (doubles as number) + (triples as number) + (hr as number);
}

export function deriveTb(
  hits: number | null | undefined,
  doubles: number | null | undefined,
  triples: number | null | undefined,
  hr: number | null | undefined,
): number | null {
  if (!allPresent(hits, doubles, triples, hr)) {
    return null;
  }
  return (hits as number) + (doubles as number) + 2 * (triples as number) + 3 * (hr as number);
}

export function deriveSbPct(
  sb: number | null | undefined,
  cs: number | null | undefined,
): number | null {
  if (!allPresent(sb, cs)) {
    return null;
  }
  const denom = (sb as number) + (cs as number);
  return denom > 0 ? (sb as number) / denom : null;
}

export function deriveTc(
  po: number | null | undefined,
  a: number | null | undefined,
  e: number | null | undefined,
): number | null {
  if (!allPresent(po, a, e)) {
    return null;
  }
  return (po as number) + (a as number) + (e as number);
}

export function deriveRf(
  po: number | null | undefined,
  a: number | null | undefined,
  inn: number | null | undefined,
): number | null {
  if (!allPresent(po, a, inn) || (inn as number) <= 0) {
    return null;
  }
  return (9 * ((po as number) + (a as number))) / (inn as number);
}

function deriveGoAo(
  go: number | null | undefined,
  ao: number | null | undefined,
): number | null {
  if (!allPresent(go, ao) || (ao as number) <= 0) {
    return null;
  }
  return (go as number) / (ao as number);
}

function deriveRate(count: number | null | undefined, opportunities: number | null | undefined): number | null {
  if (!allPresent(count, opportunities) || (opportunities as number) <= 0) {
    return null;
  }
  return (count as number) / (opportunities as number);
}

function derivePerNine(count: number | null | undefined, ip: number | null | undefined): number | null {
  if (!allPresent(count, ip) || (ip as number) <= 0) {
    return null;
  }
  return ((count as number) * 9) / (ip as number);
}

export function withHittingIdentities(row: HittingSeason): HittingSeason {
  const singles = row.singles ?? deriveSingles(row.h, row.doubles, row.triples, row.hr);
  const xbh = row.xbh ?? deriveXbh(row.doubles, row.triples, row.hr);
  const tb = row.tb ?? deriveTb(row.h, row.doubles, row.triples, row.hr);
  const sbPct = row.sb_pct ?? deriveSbPct(row.sb, row.cs);
  const goAo = row.go_ao ?? deriveGoAo(row.go, row.ao);
  const iso = row.iso ?? (allPresent(row.slg, row.avg) ? (row.slg as number) - (row.avg as number) : null);
  const babip = row.babip ?? (
    allPresent(row.h, row.hr, row.ab, row.so, row.sf)
      ? (() => {
        const denom = (row.ab as number) - (row.so as number) - (row.hr as number) + (row.sf as number);
        return denom > 0 ? ((row.h as number) - (row.hr as number)) / denom : null;
      })()
      : null
  );
  return {
    ...row,
    singles,
    xbh,
    tb,
    sb_pct: sbPct,
    go_ao: goAo,
    iso,
    babip,
    k_pct: row.k_pct ?? deriveRate(row.so, row.pa),
    bb_pct: row.bb_pct ?? deriveRate(row.bb, row.pa),
  };
}

export function withPitchingIdentities(row: PitchingSeason): PitchingSeason {
  const svo = row.svo ?? (allPresent(row.sv, row.bs) ? (row.sv as number) + (row.bs as number) : null);
  const wpct = row.wpct ?? (
    allPresent(row.w, row.l) && ((row.w as number) + (row.l as number)) > 0
      ? (row.w as number) / ((row.w as number) + (row.l as number))
      : null
  );
  return {
    ...row,
    svo,
    wpct,
    sv_pct: row.sv_pct ?? deriveRate(row.sv, svo),
    uer: row.uer ?? (allPresent(row.r, row.er) ? Math.max(0, (row.r as number) - (row.er as number)) : null),
    go_ao: row.go_ao ?? deriveGoAo(row.go, row.ao),
    k9: row.k9 ?? derivePerNine(row.so, row.ip),
    bb9: row.bb9 ?? derivePerNine(row.bb, row.ip),
    h9: row.h9 ?? derivePerNine(row.h, row.ip),
    hr9: row.hr9 ?? derivePerNine(row.hr, row.ip),
    k_bb: row.k_bb ?? (allPresent(row.so, row.bb) && (row.bb as number) > 0 ? (row.so as number) / (row.bb as number) : null),
    k_pct: row.k_pct ?? deriveRate(row.so, row.bf),
    bb_pct: row.bb_pct ?? deriveRate(row.bb, row.bf),
    i_gs: row.i_gs ?? (allPresent(row.ip, row.gs) && (row.gs as number) > 0 ? (row.ip as number) / (row.gs as number) : null),
  };
}

export function withFieldingIdentities(row: FieldingSeason): FieldingSeason {
  const ofPos = new Set(["OF", "LF", "CF", "RF"]);
  const ofa = row.ofa ?? (row.a !== null && ofPos.has(row.pos.toUpperCase()) ? row.a : null);
  return {
    ...row,
    ofa,
    tc: row.tc ?? deriveTc(row.po, row.a, row.e),
    rf: row.rf ?? deriveRf(row.po, row.a, row.inn),
    cs_pct: row.cs_pct ?? deriveSbPct(row.cs, row.sb),
  };
}

export function fieldingSignal(row: Pick<FieldingSeason, "pos" | "po" | "a" | "e" | "fpct"> | null | undefined): string {
  if (!row) {
    return "";
  }
  if (row.fpct !== null && row.fpct !== undefined) {
    return `${formatFpct(row.fpct)} FPCT`;
  }
  if (row.po !== null || row.a !== null || row.e !== null) {
    return `${formatCount(row.po)}-${formatCount(row.a)}-${formatCount(row.e)}`;
  }
  return row.pos || "";
}

export function hittingLine(row: HittingSeason): string {
  return `${formatAvg(row.avg)} AVG · ${formatOps(row.ops)} OPS · ${formatCount(row.hr)} HR`;
}

export function directoryLine(row: PlayerListItem): string {
  const off = text(row.line);
  const def = text(row.fielding_line);
  if (off && def) {
    return `${off} · ${def}`;
  }
  return off || def;
}

export function pitchingLine(row: PitchingSeason): string {
  return `${formatEra(row.era)} ERA · ${formatWhip(row.whip)} WHIP · ${formatCount(row.so)} K`;
}

function joinBits(bits: Array<string | null>): string {
  return bits.filter((bit): bit is string => Boolean(bit)).join(" · ");
}

export function hittingCountingLine(row: HittingSeason): string {
  return joinBits([
    row.g !== null ? `${formatCount(row.g)} G` : null,
    row.pa !== null ? `${formatCount(row.pa)} PA` : null,
    row.h !== null ? `${formatCount(row.h)} H` : null,
    row.hr !== null ? `${formatCount(row.hr)} HR` : null,
    row.rbi !== null ? `${formatCount(row.rbi)} RBI` : null,
    row.sb !== null ? `${formatCount(row.sb)} SB` : null,
    row.war !== null ? `${formatWar(row.war)} WAR` : null,
  ]);
}

export function hittingRatesLine(row: HittingSeason): string {
  return joinBits([
    `${formatAvg(row.avg)} AVG`,
    `${formatAvg(row.obp)} OBP`,
    `${formatAvg(row.slg)} SLG`,
    `${formatOps(row.ops)} OPS`,
  ]);
}

export function pitchingCountingLine(row: PitchingSeason): string {
  return joinBits([
    row.g !== null ? `${formatCount(row.g)} G` : null,
    row.ip !== null ? `${formatIp(row.ip)} IP` : null,
    row.so !== null ? `${formatCount(row.so)} K` : null,
    row.bb !== null ? `${formatCount(row.bb)} BB` : null,
    row.w !== null || row.l !== null ? formatWl(row) : null,
    row.war !== null ? `${formatWar(row.war)} WAR` : null,
  ]);
}

export function pitchingRatesLine(row: PitchingSeason): string {
  return joinBits([`${formatEra(row.era)} ERA`, `${formatWhip(row.whip)} WHIP`]);
}

export function isApproxWar(source: string | null | undefined): boolean {
  return String(source || "").trim().toLowerCase() === "approx";
}

export function selectedYearMissing(
  health: Health,
  season: number,
  hasLine: boolean,
): boolean {
  if (hasLine) {
    return false;
  }
  return Boolean(health.current_season_missing && season === health.active_season);
}

export function isPitcherPosition(position: string): boolean {
  const key = position.trim().toUpperCase();
  return key === "P" || key === "SP" || key === "RP" || key === "CL";
}

export function seasonWindowYears(health: Health): number[] {
  const start = Number(health.season_window?.start) || DEFAULT_ACTIVE_SEASON - 2;
  const end = Number(health.season_window?.end) || DEFAULT_ACTIVE_SEASON;
  const years: number[] = [];
  for (let year = start; year <= end; year += 1) {
    years.push(year);
  }
  return years;
}

export function defaultDirectorySeason(health: Health, seasons: number[] = []): number {
  const active = Number(health.active_season) || DEFAULT_ACTIVE_SEASON;
  if (health.current_season_missing) {
    const present = seasons.filter((year) => Number.isFinite(year) && year < active);
    return present.length ? Math.max(...present) : active - 1;
  }
  return active;
}

export function playerQualifies(
  item: Pick<PlayerListItem, "pa" | "ip">,
  minPa = MIN_PA,
  minIp = MIN_IP,
): boolean {
  const pa = item.pa ?? 0;
  const ip = item.ip ?? 0;
  return pa >= minPa || ip >= minIp;
}

export function dropCurrentSeasonLines<T extends { season: number }>(
  rows: T[],
  health: Health,
): T[] {
  if (!health.current_season_missing) {
    return rows;
  }
  const active = Number(health.active_season) || DEFAULT_ACTIVE_SEASON;
  return rows.filter((row) => Number(row.season) !== active);
}

export function cardChipFromCards(
  cards: FantasyCard[],
  playerId: string,
): PlayerCardChip | null {
  const wanted = String(playerId || "").trim();
  if (!wanted) {
    return null;
  }
  const card = cards.find((row) => text(asRecord(row.player).player_id) === wanted);
  if (!card) {
    return null;
  }
  return {
    recommendation_type: text(card.recommendation_type),
    label: recommendationLabel(card.recommendation_type),
    reason: cardReason(card),
  };
}

function parseIdentity(value: unknown, fallbackId = ""): PlayerIdentity {
  const raw = asRecord(value);
  return {
    player_id: text(raw.player_id || raw.id || fallbackId),
    name: text(raw.name || raw.display_name),
    position: text(raw.position || raw.pos),
    team: text(raw.team),
  };
}

function parseHittingSeason(value: unknown): HittingSeason | null {
  const raw = asRecord(value);
  const season = num(raw.season);
  if (season === null) {
    return null;
  }
  return withHittingIdentities({
    season,
    g: num(raw.g ?? raw.games),
    pa: num(raw.pa),
    ab: num(raw.ab),
    r: num(raw.r ?? raw.runs),
    h: num(raw.h ?? raw.hits),
    doubles: num(raw.doubles ?? raw["2b"]),
    triples: num(raw.triples ?? raw["3b"]),
    hr: num(raw.hr),
    rbi: num(raw.rbi),
    sb: num(raw.sb),
    bb: num(raw.bb),
    so: num(raw.so),
    avg: num(raw.avg),
    obp: num(raw.obp),
    slg: num(raw.slg),
    ops: num(raw.ops),
    woba: num(raw.woba),
    war: num(raw.war),
    war_source: text(raw.war_source),
    singles: num(raw.singles ?? raw["1b"]),
    xbh: num(raw.xbh),
    tb: num(raw.tb),
    cs: num(raw.cs),
    sb_pct: num(raw.sb_pct),
    hbp: num(raw.hbp),
    sh: num(raw.sh),
    sf: num(raw.sf),
    gidp: num(raw.gidp),
    ibb: num(raw.ibb),
    lob: num(raw.lob),
    roe: num(raw.roe),
    gsh: num(raw.gsh),
    go: num(raw.go),
    ao: num(raw.ao),
    go_ao: num(raw.go_ao),
    iso: num(raw.iso),
    babip: num(raw.babip),
    k_pct: num(raw.k_pct),
    bb_pct: num(raw.bb_pct),
  });
}

function parsePitchingSeason(value: unknown): PitchingSeason | null {
  const raw = asRecord(value);
  const season = num(raw.season);
  if (season === null) {
    return null;
  }
  return withPitchingIdentities({
    season,
    g: num(raw.g ?? raw.games),
    gs: num(raw.gs),
    ip: num(raw.ip),
    w: num(raw.w ?? raw.wins),
    l: num(raw.l ?? raw.losses),
    sv: num(raw.sv ?? raw.saves),
    so: num(raw.so ?? raw.pitching_so),
    bb: num(raw.bb ?? raw.pitching_bb),
    er: num(raw.er),
    era: num(raw.era),
    whip: num(raw.whip),
    fip: num(raw.fip),
    war: num(raw.war),
    war_source: text(raw.war_source),
    h: num(raw.h ?? raw.pitching_hits),
    hr: num(raw.hr ?? raw.pitching_hr),
    r: num(raw.r ?? raw.pitching_r),
    uer: num(raw.uer),
    cg: num(raw.cg),
    sho: num(raw.sho),
    hld: num(raw.hld),
    bs: num(raw.bs),
    svo: num(raw.svo),
    sv_pct: num(raw.sv_pct),
    qs: num(raw.qs),
    gf: num(raw.gf),
    bk: num(raw.bk),
    wp: num(raw.wp),
    np: num(raw.np),
    pk: num(raw.pk),
    ir: num(raw.ir),
    bf: num(raw.bf),
    go: num(raw.go ?? raw.pitching_go),
    ao: num(raw.ao ?? raw.pitching_ao),
    go_ao: num(raw.go_ao ?? raw.pitching_go_ao),
    wpct: num(raw.wpct),
    k9: num(raw.k9),
    bb9: num(raw.bb9),
    h9: num(raw.h9),
    hr9: num(raw.hr9),
    k_bb: num(raw.k_bb),
    k_pct: num(raw.k_pct ?? raw.pitching_k_pct),
    bb_pct: num(raw.bb_pct ?? raw.pitching_bb_pct),
    i_gs: num(raw.i_gs),
  });
}

function parseFieldingSeason(value: unknown, fallbackSeason?: number): FieldingSeason | null {
  const raw = asRecord(value);
  const season = num(raw.season) ?? fallbackSeason ?? null;
  if (season === null) {
    return null;
  }
  const line: FieldingSeason = withFieldingIdentities({
    season,
    pos: text(raw.pos || raw.position || raw.fielding_pos),
    g: num(raw.g ?? raw.fielding_g),
    gs: num(raw.gs ?? raw.fielding_gs),
    inn: num(raw.inn ?? raw.fielding_inn),
    po: num(raw.po ?? raw.putouts),
    a: num(raw.a ?? raw.assists),
    e: num(raw.e ?? raw.errors),
    dp: num(raw.dp ?? raw.double_plays),
    pb: num(raw.pb ?? raw.passed_balls),
    fpct: num(raw.fpct),
    ofa: num(raw.ofa),
    cs: num(raw.cs ?? raw.fielding_cs),
    sb: num(raw.sb ?? raw.fielding_sb),
    tp: num(raw.tp),
    tc: num(raw.tc),
    rf: num(raw.rf),
    cs_pct: num(raw.cs_pct),
  });
  const counts = [
    line.g, line.gs, line.inn, line.po, line.a, line.e, line.dp, line.pb, line.fpct,
    line.ofa ?? null, line.cs ?? null, line.tp ?? null, line.tc ?? null,
  ];
  if (!line.pos && counts.every((value) => value === null)) {
    return null;
  }
  if (counts.every((value) => value === null)) {
    return null;
  }
  return line;
}

function parseFieldingBlob(raw: Record<string, unknown>, season: number): FieldingSeason[] {
  const listed = asList(raw.fielding);
  if (listed.length) {
    return listed
      .map((item) => parseFieldingSeason({ ...asRecord(item), season }, season))
      .filter((row): row is FieldingSeason => row !== null);
  }
  const blob = raw.fielding_json;
  if (typeof blob === "string" && blob.trim()) {
    try {
      const parsed = JSON.parse(blob) as unknown;
      if (Array.isArray(parsed)) {
        return parsed
          .map((item) => parseFieldingSeason({ ...asRecord(item), season }, season))
          .filter((row): row is FieldingSeason => row !== null);
      }
    } catch {
      // Honest omit when the blob is not JSON.
    }
  }
  const single = parseFieldingSeason(
    {
      season,
      pos: raw.fielding_pos,
      g: raw.fielding_g,
      gs: raw.fielding_gs,
      inn: raw.fielding_inn,
      po: raw.putouts ?? raw.po,
      a: raw.assists ?? raw.a,
      e: raw.errors ?? raw.e,
      dp: raw.double_plays ?? raw.dp,
      pb: raw.passed_balls ?? raw.pb,
      fpct: raw.fpct,
      ofa: raw.ofa,
      cs: raw.fielding_cs,
      sb: raw.fielding_sb,
      tp: raw.tp,
      tc: raw.tc,
      rf: raw.rf,
      cs_pct: raw.cs_pct,
    },
    season,
  );
  return single ? [single] : [];
}

function parseHittingGame(value: unknown): HittingGame | null {
  const raw = asRecord(value);
  const date = text(raw.date || raw.game_date);
  if (!date) {
    return null;
  }
  return {
    date,
    opponent: text(raw.opponent || raw.opp),
    season: num(raw.season) ?? (Number(date.slice(0, 4)) || 0),
    ab: num(raw.ab),
    r: num(raw.r ?? raw.runs),
    h: num(raw.h ?? raw.hits),
    doubles: num(raw.doubles ?? raw["2b"]),
    triples: num(raw.triples ?? raw["3b"]),
    hr: num(raw.hr),
    rbi: num(raw.rbi),
    sb: num(raw.sb),
    bb: num(raw.bb),
    so: num(raw.so),
  };
}

function parsePitchingGame(value: unknown): PitchingGame | null {
  const raw = asRecord(value);
  const date = text(raw.date || raw.game_date);
  if (!date) {
    return null;
  }
  return {
    date,
    opponent: text(raw.opponent || raw.opp),
    season: num(raw.season) ?? (Number(date.slice(0, 4)) || 0),
    ip: num(raw.ip),
    h: num(raw.h ?? raw.hits),
    er: num(raw.er),
    hr: num(raw.hr ?? raw.pitching_hr),
    bb: num(raw.bb ?? raw.pitching_bb),
    so: num(raw.so ?? raw.pitching_so),
    gs: num(raw.gs ?? raw.games_started),
    decision: text(raw.decision) || null,
  };
}

function seasonLooksLikePitching(raw: Record<string, unknown>): boolean {
  const type = text(raw.player_type).toLowerCase();
  if (type === "pitcher" || type === "pitching") {
    return true;
  }
  return num(raw.ip) !== null || num(raw.era) !== null || num(raw.whip) !== null;
}

function seasonLooksLikeHitting(raw: Record<string, unknown>): boolean {
  const type = text(raw.player_type).toLowerCase();
  if (type === "batter" || type === "hitting" || type === "hitter") {
    return true;
  }
  return num(raw.pa) !== null || num(raw.avg) !== null || num(raw.ops) !== null;
}

function expandPublishedSeasons(seasons: unknown[]): {
  hitting: HittingSeason[];
  pitching: PitchingSeason[];
  fielding: FieldingSeason[];
} {
  const hitting: HittingSeason[] = [];
  const pitching: PitchingSeason[] = [];
  const fielding: FieldingSeason[] = [];
  for (const row of seasons) {
    const raw = asRecord(row);
    const hit = seasonLooksLikeHitting(raw);
    const pitch = seasonLooksLikePitching(raw);
    if (hit || !pitch) {
      const parsed = parseHittingSeason(row);
      if (parsed) {
        hitting.push(parsed);
      }
    }
    if (pitch) {
      const parsed = parsePitchingSeason(row);
      if (parsed) {
        pitching.push(parsed);
      }
    }
    const season = num(raw.season);
    if (season !== null) {
      fielding.push(...parseFieldingBlob(raw, season));
    }
  }
  return { hitting, pitching, fielding };
}

export function parsePlayerDetail(
  payload: unknown,
  source: "api" | "stub" = "api",
): PlayerDetail | null {
  if (payload === null || payload === undefined) {
    return null;
  }
  const raw = asRecord(payload);
  const nested = raw.player !== undefined ? raw.player : raw;
  const player = parseIdentity(nested, text(raw.player_id));
  if (!player.player_id) {
    return null;
  }
  const nestedRecord = asRecord(nested);
  const published = asList(nestedRecord.seasons || raw.seasons);
  const fromPublished = published.length ? expandPublishedSeasons(published) : null;
  const gamesRaw = asRecord(raw.recent_games || raw.game_log || raw.games || nestedRecord.recent_games);
  const cardRaw = raw.card === undefined || raw.card === null ? null : asRecord(raw.card);
  const fallbackSeason = fromPublished?.hitting[0]?.season ?? fromPublished?.pitching[0]?.season ?? undefined;
  const fielding = fromPublished
    ? fromPublished.fielding
    : [
        ...asList(raw.fielding).map((item) => parseFieldingSeason(item, fallbackSeason)),
        ...asList(nestedRecord.fielding).map((item) => parseFieldingSeason(item, fallbackSeason)),
      ].filter((row): row is FieldingSeason => row !== null);
  return {
    player,
    hitting: fromPublished
      ? fromPublished.hitting
      : asList(raw.hitting || raw.batting)
          .map(parseHittingSeason)
          .filter((row): row is HittingSeason => row !== null),
    pitching: fromPublished
      ? fromPublished.pitching
      : asList(raw.pitching)
          .map(parsePitchingSeason)
          .filter((row): row is PitchingSeason => row !== null),
    fielding,
    recent_games: {
      hitting: asList(gamesRaw.hitting)
        .map(parseHittingGame)
        .filter((row): row is HittingGame => row !== null),
      pitching: asList(gamesRaw.pitching)
        .map(parsePitchingGame)
        .filter((row): row is PitchingGame => row !== null),
    },
    card: cardRaw
      ? {
          recommendation_type: text(cardRaw.recommendation_type),
          label: text(cardRaw.label) || recommendationLabel(text(cardRaw.recommendation_type)),
          reason: text(cardRaw.reason),
        }
      : null,
    source,
  };
}

function parseListItem(value: unknown, fallbackSeason: number): PlayerListItem | null {
  const raw = asRecord(value);
  const player = parseIdentity(raw.player !== undefined ? raw.player : raw, text(raw.player_id));
  if (!player.player_id) {
    return null;
  }
  const season = num(raw.season) ?? fallbackSeason;
  const side: PlayerSide = text(raw.side).toLowerCase() === "pitching" ? "pitching" : "hitting";
  const hitting = parseHittingSeason({ ...raw, season });
  const pitching = parsePitchingSeason({ ...raw, season });
  const pa = num(raw.pa);
  const ip = num(raw.ip);
  const war = num(raw.war);
  const line = text(raw.line)
    || (side === "pitching" && pitching ? pitchingLine(pitching) : "")
    || (hitting ? hittingLine(hitting) : "");
  return {
    player_id: player.player_id,
    name: player.name,
    position: player.position,
    team: player.team,
    season,
    side,
    pa,
    ip,
    war,
    edge: num(raw.edge ?? raw.vs_replacement),
    line,
    fpct: num(raw.fpct),
    fielding_line: fieldingSignal(parseFieldingSeason({ ...raw, season }, season)),
  };
}

export function parsePlayersList(
  payload: unknown,
  fallbackSeason: number,
): PlayerListItem[] {
  const raw = Array.isArray(payload) ? payload : asList(asRecord(payload).players || asRecord(payload).items);
  return raw
    .map((row) => {
      const rec = asRecord(row);
      const seasons = asList(rec.seasons);
      if (!seasons.length) {
        return parseListItem(row, fallbackSeason);
      }
      const match =
        seasons.find((season) => num(asRecord(season).season) === fallbackSeason) ||
        (Number.isFinite(fallbackSeason) ? null : seasons[0]);
      if (!match) {
        return null;
      }
      const line = asRecord(match);
      return parseListItem(
        {
          player_id: rec.player_id,
          name: rec.name,
          position: rec.position || line.position,
          team: rec.team || line.team,
          ...line,
        },
        fallbackSeason,
      );
    })
    .filter((row): row is PlayerListItem => row !== null);
}

export function listItemFromDetail(
  detail: PlayerDetail,
  season: number,
  edge: number | null = null,
): PlayerListItem | null {
  const hitting = detail.hitting.find((row) => row.season === season) || null;
  const pitching = detail.pitching.find((row) => row.season === season) || null;
  if (!hitting && !pitching) {
    return null;
  }
  const preferPitching = isPitcherPosition(detail.player.position);
  const side: PlayerSide = preferPitching && pitching ? "pitching" : hitting ? "hitting" : "pitching";
  const war = side === "pitching" ? (pitching?.war ?? null) : (hitting?.war ?? null);
  const fielding = detail.fielding.find((row) => row.season === season) || null;
  const off = side === "pitching" && pitching ? pitchingLine(pitching) : hitting ? hittingLine(hitting) : "";
  const def = fieldingSignal(fielding);
  return {
    player_id: detail.player.player_id,
    name: detail.player.name,
    position: detail.player.position,
    team: detail.player.team,
    season,
    side,
    pa: hitting?.pa ?? null,
    ip: pitching?.ip ?? null,
    war,
    edge,
    line: off,
    fpct: fielding?.fpct ?? null,
    fielding_line: def,
  };
}

function edgeFromCards(cards: FantasyCard[], playerId: string): number | null {
  const card = cards.find((row) => text(asRecord(row.player).player_id) === playerId);
  if (!card) {
    return null;
  }
  return num(asRecord(card.edge).vs_replacement);
}

export function filterStubPlayers(
  records: StubPlayerRecord[],
  cards: FantasyCard[],
  health: Health,
  query: PlayersQuery = {},
): PlayersResponse {
  const season = query.season ?? defaultDirectorySeason(health, seasonWindowYears(health));
  const sort = text(query.sort) || DEFAULT_PLAYER_SORT;
  const minPa = query.min_pa ?? MIN_PA;
  const minIp = query.min_ip ?? MIN_IP;
  if (health.current_season_missing && season === health.active_season) {
    return { players: [], season, sort, source: "stub" };
  }

  const q = text(query.q).toLowerCase();
  const position = text(query.position).toUpperCase();
  const team = text(query.team).toUpperCase();

  const players = records
    .map((record) => {
      const detail = parsePlayerDetail(record, "stub");
      if (!detail) {
        return null;
      }
      return listItemFromDetail(detail, season, edgeFromCards(cards, detail.player.player_id));
    })
    .filter((row): row is PlayerListItem => row !== null)
    .filter((row) => playerQualifies(row, minPa, minIp))
    .filter((row) => {
      if (position && row.position.toUpperCase() !== position) {
        return false;
      }
      if (team && row.team.toUpperCase() !== team) {
        return false;
      }
      if (!q) {
        return true;
      }
      const hay = `${row.name} ${row.team} ${row.position} ${row.player_id}`.toLowerCase();
      return hay.includes(q);
    })
    .sort((left, right) => {
      const key = sort === "edge" ? "edge" : "war";
      return (right[key] ?? Number.NEGATIVE_INFINITY) - (left[key] ?? Number.NEGATIVE_INFINITY);
    });

  return { players, season, sort, source: "stub" };
}

export function stubPlayerDetail(
  records: StubPlayerRecord[],
  cards: FantasyCard[],
  health: Health,
  playerId: string,
): PlayerDetail | null {
  const record = records.find((row) => text(asRecord(row.player).player_id) === String(playerId || "").trim());
  if (!record) {
    return null;
  }
  const parsed = parsePlayerDetail(record, "stub");
  if (!parsed) {
    return null;
  }
  return {
    ...parsed,
    hitting: dropCurrentSeasonLines(parsed.hitting, health),
    pitching: dropCurrentSeasonLines(parsed.pitching, health),
    fielding: dropCurrentSeasonLines(parsed.fielding, health),
    recent_games: {
      hitting: dropCurrentSeasonLines(parsed.recent_games.hitting, health),
      pitching: dropCurrentSeasonLines(parsed.recent_games.pitching, health),
    },
    card: cardChipFromCards(cards, parsed.player.player_id),
    source: "stub",
  };
}

export function honestyFilterDetail(detail: PlayerDetail, health: Health): PlayerDetail {
  return {
    ...detail,
    hitting: dropCurrentSeasonLines(detail.hitting, health),
    pitching: dropCurrentSeasonLines(detail.pitching, health),
    fielding: dropCurrentSeasonLines(detail.fielding, health),
    recent_games: {
      hitting: dropCurrentSeasonLines(detail.recent_games.hitting, health),
      pitching: dropCurrentSeasonLines(detail.recent_games.pitching, health),
    },
  };
}

export function buildPlayersQueryString(query: PlayersQuery = {}): string {
  const params = new URLSearchParams();
  if (query.season !== undefined) {
    params.set("season", String(query.season));
  }
  if (query.sort) {
    params.set("sort", String(query.sort));
  }
  if (query.min_pa !== undefined) {
    params.set("min_pa", String(query.min_pa));
  }
  if (query.q) {
    params.set("q", String(query.q));
  }
  if (query.position) {
    params.set("position", String(query.position));
  }
  if (query.team) {
    params.set("team", String(query.team));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}
