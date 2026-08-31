import {
  formatAvg,
  formatCount,
  formatEra,
  formatIp,
  formatOps,
  formatWar,
  formatWhip,
  playerQualifies,
  type HittingSeason,
  type PitchingSeason,
  type PlayerDetail,
  type PlayerListItem,
} from "@bos/api-client";

export const COMPARE_MAX = 4;
export const COMPARE_MIN = 2;
export const COMPARE_STORAGE_KEY = "bos.compare.selection";
export const EMPTY_CELL = "—";

export type CompareMode = "players" | "teams";

export type CompareQuery = {
  mode: CompareMode;
  season: number;
  ids: string[];
};

export type CompareSearchInput = {
  mode?: string | string[];
  season?: string | string[];
  ids?: string | string[];
};

export type StoredCompare = {
  season: number;
  ids: string[];
};

export type StorageLike = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

export type CompareColumn = {
  player_id: string;
  name: string;
  position: string;
  team: string;
  hitting: HittingSeason | null;
  pitching: PitchingSeason | null;
  found: boolean;
};

export type CompareDirection = "higher" | "lower";

export type CompareStatDef = {
  key: string;
  label: string;
  block: "hitting" | "pitching";
  direction: CompareDirection;
  read: (column: CompareColumn) => number | null;
  format: (value: number | null | undefined) => string;
};

export type CompareRowView = {
  key: string;
  label: string;
  block: "hitting" | "pitching";
  display: string[];
  best: number[];
};

function firstParam(value: string | string[] | undefined): string {
  if (Array.isArray(value)) {
    return String(value[0] ?? "").trim();
  }
  return String(value ?? "").trim();
}

export function parseCompareIds(raw: string | string[] | undefined): string[] {
  const text = Array.isArray(raw) ? raw.join(",") : String(raw ?? "");
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const part of text.split(",")) {
    const id = part.trim();
    if (!id || seen.has(id)) {
      continue;
    }
    seen.add(id);
    ids.push(id);
    if (ids.length >= COMPARE_MAX) {
      break;
    }
  }
  return ids;
}

export function parseCompareMode(raw: string | string[] | undefined): CompareMode {
  return firstParam(raw).toLowerCase() === "teams" ? "teams" : "players";
}

export function parseCompareSeason(
  raw: string | string[] | undefined,
  fallback: number,
): number {
  const year = Number(firstParam(raw));
  return Number.isFinite(year) && year > 1800 ? year : fallback;
}

export function clampSeason(year: number, years: number[], fallback: number): number {
  if (years.includes(year)) {
    return year;
  }
  return fallback;
}

export function parseCompareQuery(
  raw: CompareSearchInput,
  fallbackSeason: number,
  years: number[] = [],
): CompareQuery {
  return {
    mode: "players",
    season: clampSeason(parseCompareSeason(raw.season, fallbackSeason), years, fallbackSeason),
    ids: parseCompareIds(raw.ids),
  };
}

export function buildComparePath(query: {
  season: number;
  ids: string[];
  mode?: CompareMode;
}): string {
  const params = [`mode=players`, `season=${query.season}`];
  if (query.ids.length) {
    params.push(`ids=${query.ids.map((id) => encodeURIComponent(id)).join(",")}`);
  }
  return `/compare?${params.join("&")}`;
}

export function appendCompareId(ids: string[], playerId: string): string[] {
  const id = String(playerId || "").trim();
  if (!id || ids.includes(id) || ids.length >= COMPARE_MAX) {
    return ids.slice(0, COMPARE_MAX);
  }
  return [...ids, id];
}

export function removeCompareId(ids: string[], playerId: string): string[] {
  return ids.filter((id) => id !== playerId);
}

export function slotIds(ids: string[]): Array<string | null> {
  const slots: Array<string | null> = [null, null, null, null];
  ids.slice(0, COMPARE_MAX).forEach((id, index) => {
    slots[index] = id;
  });
  return slots;
}

function browserStorage(): StorageLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function readStoredCompare(storage: StorageLike | null = browserStorage()): StoredCompare | null {
  if (!storage) {
    return null;
  }
  try {
    const raw = storage.getItem(COMPARE_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    const parsed = JSON.parse(raw) as Partial<StoredCompare>;
    const season = Number(parsed.season);
    const ids = parseCompareIds(Array.isArray(parsed.ids) ? parsed.ids.join(",") : "");
    if (!Number.isFinite(season) || season <= 1800) {
      return null;
    }
    return { season, ids };
  } catch {
    return null;
  }
}

export function writeStoredCompare(
  selection: StoredCompare,
  storage: StorageLike | null = browserStorage(),
): void {
  if (!storage) {
    return;
  }
  storage.setItem(
    COMPARE_STORAGE_KEY,
    JSON.stringify({
      season: selection.season,
      ids: selection.ids.slice(0, COMPARE_MAX),
    }),
  );
}

export function compareHrefForPlayer(
  playerId: string,
  season: number,
  storage: StorageLike | null = browserStorage(),
): string {
  const stored = readStoredCompare(storage);
  const ids = appendCompareId(stored?.ids ?? [], playerId);
  const year = stored?.season ?? season;
  const next = { season: year, ids };
  writeStoredCompare(next, storage);
  return buildComparePath(next);
}

export function columnFromDetail(
  playerId: string,
  detail: PlayerDetail | null,
  season: number,
  fallback?: Pick<PlayerListItem, "name" | "position" | "team"> | null,
): CompareColumn {
  if (!detail) {
    return {
      player_id: playerId,
      name: fallback?.name || playerId,
      position: fallback?.position || "",
      team: fallback?.team || "",
      hitting: null,
      pitching: null,
      found: Boolean(fallback),
    };
  }
  return {
    player_id: detail.player.player_id,
    name: detail.player.name,
    position: detail.player.position,
    team: detail.player.team,
    hitting: detail.hitting.find((row) => row.season === season) ?? null,
    pitching: detail.pitching.find((row) => row.season === season) ?? null,
    found: true,
  };
}

export const HITTING_STATS: CompareStatDef[] = [
  { key: "g", label: "G", block: "hitting", direction: "higher", read: (col) => col.hitting?.g ?? null, format: formatCount },
  { key: "pa", label: "PA", block: "hitting", direction: "higher", read: (col) => col.hitting?.pa ?? null, format: formatCount },
  { key: "h", label: "H", block: "hitting", direction: "higher", read: (col) => col.hitting?.h ?? null, format: formatCount },
  { key: "hr", label: "HR", block: "hitting", direction: "higher", read: (col) => col.hitting?.hr ?? null, format: formatCount },
  { key: "r", label: "R", block: "hitting", direction: "higher", read: (col) => col.hitting?.r ?? null, format: formatCount },
  { key: "rbi", label: "RBI", block: "hitting", direction: "higher", read: (col) => col.hitting?.rbi ?? null, format: formatCount },
  { key: "sb", label: "SB", block: "hitting", direction: "higher", read: (col) => col.hitting?.sb ?? null, format: formatCount },
  { key: "bb", label: "BB", block: "hitting", direction: "higher", read: (col) => col.hitting?.bb ?? null, format: formatCount },
  { key: "so", label: "K", block: "hitting", direction: "higher", read: (col) => col.hitting?.so ?? null, format: formatCount },
  { key: "avg", label: "AVG", block: "hitting", direction: "higher", read: (col) => col.hitting?.avg ?? null, format: formatAvg },
  { key: "obp", label: "OBP", block: "hitting", direction: "higher", read: (col) => col.hitting?.obp ?? null, format: formatAvg },
  { key: "slg", label: "SLG", block: "hitting", direction: "higher", read: (col) => col.hitting?.slg ?? null, format: formatAvg },
  { key: "ops", label: "OPS", block: "hitting", direction: "higher", read: (col) => col.hitting?.ops ?? null, format: formatOps },
  { key: "war", label: "WAR", block: "hitting", direction: "higher", read: (col) => col.hitting?.war ?? null, format: formatWar },
];

export const PITCHING_STATS: CompareStatDef[] = [
  { key: "g", label: "G", block: "pitching", direction: "higher", read: (col) => col.pitching?.g ?? null, format: formatCount },
  { key: "gs", label: "GS", block: "pitching", direction: "higher", read: (col) => col.pitching?.gs ?? null, format: formatCount },
  { key: "ip", label: "IP", block: "pitching", direction: "higher", read: (col) => col.pitching?.ip ?? null, format: formatIp },
  { key: "w", label: "W", block: "pitching", direction: "higher", read: (col) => col.pitching?.w ?? null, format: formatCount },
  { key: "l", label: "L", block: "pitching", direction: "higher", read: (col) => col.pitching?.l ?? null, format: formatCount },
  { key: "sv", label: "SV", block: "pitching", direction: "higher", read: (col) => col.pitching?.sv ?? null, format: formatCount },
  { key: "so", label: "K", block: "pitching", direction: "higher", read: (col) => col.pitching?.so ?? null, format: formatCount },
  { key: "bb", label: "BB", block: "pitching", direction: "higher", read: (col) => col.pitching?.bb ?? null, format: formatCount },
  { key: "era", label: "ERA", block: "pitching", direction: "lower", read: (col) => col.pitching?.era ?? null, format: formatEra },
  { key: "whip", label: "WHIP", block: "pitching", direction: "lower", read: (col) => col.pitching?.whip ?? null, format: formatWhip },
  { key: "war", label: "WAR", block: "pitching", direction: "higher", read: (col) => col.pitching?.war ?? null, format: formatWar },
];

export function bestIndexes(
  values: Array<number | null>,
  direction: CompareDirection,
): number[] {
  const scored = values
    .map((value, index) => ({ value, index }))
    .filter((row): row is { value: number; index: number } => (
      row.value !== null && Number.isFinite(row.value)
    ));
  if (!scored.length) {
    return [];
  }
  const best = scored.reduce((winner, row) => {
    if (direction === "lower") {
      return row.value < winner ? row.value : winner;
    }
    return row.value > winner ? row.value : winner;
  }, scored[0].value);
  return scored.filter((row) => row.value === best).map((row) => row.index);
}

export function buildCompareRows(columns: CompareColumn[]): CompareRowView[] {
  const hasHitting = columns.some((column) => column.hitting);
  const hasPitching = columns.some((column) => column.pitching);
  const defs = [
    ...(hasHitting ? HITTING_STATS : []),
    ...(hasPitching ? PITCHING_STATS : []),
  ];
  return defs.map((def) => {
    const values = columns.map(def.read);
    return {
      key: `${def.block}-${def.key}`,
      label: def.label,
      block: def.block,
      display: values.map((value) => {
        const text = def.format(value);
        return text || EMPTY_CELL;
      }),
      best: bestIndexes(values, def.direction),
    };
  });
}

export function filterSlotCandidates(
  pool: PlayerListItem[],
  query: string,
  selectedIds: string[],
  limit = 12,
): PlayerListItem[] {
  const selected = new Set(selectedIds);
  const needle = query.trim().toLowerCase();
  return pool
    .filter((row) => !selected.has(row.player_id) && playerQualifies(row))
    .filter((row) => {
      if (!needle) {
        return true;
      }
      const hay = `${row.name} ${row.team} ${row.position} ${row.player_id}`.toLowerCase();
      return hay.includes(needle);
    })
    .sort((left, right) => {
      const war = (right.war ?? Number.NEGATIVE_INFINITY) - (left.war ?? Number.NEGATIVE_INFINITY);
      if (war !== 0) {
        return war;
      }
      return left.name.localeCompare(right.name);
    })
    .slice(0, limit);
}
