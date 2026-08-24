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
  hr: number | null;
  rbi: number | null;
  sb: number | null;
  bb: number | null;
  so: number | null;
  avg: number | null;
  obp: number | null;
  slg: number | null;
  ops: number | null;
  war: number | null;
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
  era: number | null;
  whip: number | null;
  war: number | null;
};

export type HittingGame = {
  date: string;
  opponent: string;
  season: number;
  ab: number | null;
  r: number | null;
  h: number | null;
  hr: number | null;
  rbi: number | null;
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
  bb: number | null;
  so: number | null;
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

export function hittingLine(row: HittingSeason): string {
  return `${formatAvg(row.avg)} AVG · ${formatOps(row.ops)} OPS · ${formatCount(row.hr)} HR`;
}

export function pitchingLine(row: PitchingSeason): string {
  return `${formatEra(row.era)} ERA · ${formatWhip(row.whip)} WHIP · ${formatCount(row.so)} K`;
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
  return {
    season,
    g: num(raw.g ?? raw.games),
    pa: num(raw.pa),
    ab: num(raw.ab),
    r: num(raw.r ?? raw.runs),
    h: num(raw.h ?? raw.hits),
    hr: num(raw.hr),
    rbi: num(raw.rbi),
    sb: num(raw.sb),
    bb: num(raw.bb),
    so: num(raw.so),
    avg: num(raw.avg),
    obp: num(raw.obp),
    slg: num(raw.slg),
    ops: num(raw.ops),
    war: num(raw.war),
  };
}

function parsePitchingSeason(value: unknown): PitchingSeason | null {
  const raw = asRecord(value);
  const season = num(raw.season);
  if (season === null) {
    return null;
  }
  return {
    season,
    g: num(raw.g ?? raw.games),
    gs: num(raw.gs),
    ip: num(raw.ip),
    w: num(raw.w ?? raw.wins),
    l: num(raw.l ?? raw.losses),
    sv: num(raw.sv ?? raw.saves),
    so: num(raw.so ?? raw.pitching_so),
    bb: num(raw.bb ?? raw.pitching_bb),
    era: num(raw.era),
    whip: num(raw.whip),
    war: num(raw.war),
  };
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
    hr: num(raw.hr),
    rbi: num(raw.rbi),
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
    bb: num(raw.bb ?? raw.pitching_bb),
    so: num(raw.so ?? raw.pitching_so),
    decision: text(raw.decision) || null,
  };
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
  const gamesRaw = asRecord(raw.recent_games || raw.game_log || raw.games);
  const cardRaw = raw.card === undefined || raw.card === null ? null : asRecord(raw.card);
  return {
    player,
    hitting: asList(raw.hitting || raw.batting)
      .map(parseHittingSeason)
      .filter((row): row is HittingSeason => row !== null),
    pitching: asList(raw.pitching)
      .map(parsePitchingSeason)
      .filter((row): row is PitchingSeason => row !== null),
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
  };
}

export function parsePlayersList(
  payload: unknown,
  fallbackSeason: number,
): PlayerListItem[] {
  const raw = Array.isArray(payload) ? payload : asList(asRecord(payload).players || asRecord(payload).items);
  return raw
    .map((row) => parseListItem(row, fallbackSeason))
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
    line: side === "pitching" && pitching ? pitchingLine(pitching) : hitting ? hittingLine(hitting) : "",
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
