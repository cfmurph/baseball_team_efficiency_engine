import {
  DEFAULT_ACTIVE_SEASON,
  RECOMMENDATION_TYPES,
  SCHEMA_VERSION,
  type ArtifactSource,
  type CardsResponse,
  type FantasyCard,
  type Health,
  type RecommendationType,
  type SeasonWindow,
  type SeasonsResponse,
} from "@bos/card-schema";

import stubCards from "./stub-cards.json" with { type: "json" };

export type ApiClientOptions = {
  /** Base URL for the #144 / #106 service. No trailing slash. */
  baseUrl?: string | null;
  fetch?: typeof fetch;
  /** QA override so stub health can raise the not-current-year banner. */
  stubCurrentSeasonMissing?: boolean;
};

export type CardQuery = {
  season?: number;
  rec?: RecommendationType | string;
};

export type PlayerQuery = {
  season?: number;
};

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
  avg?: number | null;
  woba?: number | null;
  era?: number | null;
  whip?: number | null;
  fip?: number | null;
};

export type PlayerRecord = {
  player_id: string;
  name?: string | null;
  position?: string | null;
  team?: string | null;
  seasons: PlayerSeason[];
};

export type PlayersResponse = {
  as_of: string;
  active_season: number;
  current_season_missing: boolean;
  season_window: number[];
  seasons_present: number[];
  source?: "remote" | "local" | "missing";
  current_season_missing_reason?: string | null;
  season?: number | null;
  players: PlayerRecord[];
};

export type PlayerResponse = Omit<PlayersResponse, "players"> & {
  player: PlayerRecord | null;
};

const STUB_AS_OF = "2026-08-23";

/** Local #144 process. Prefer this when `NEXT_PUBLIC_API_URL` is unset. */
export const LOCAL_V1_ORIGIN = "http://127.0.0.1:8000";

export async function probeLocalV1(
  fetcher: typeof fetch = fetch,
  origin = LOCAL_V1_ORIGIN,
): Promise<string | null> {
  const base = String(origin || "").trim().replace(/\/+$/, "");
  if (!base) {
    return null;
  }
  try {
    const response = await fetcher(`${base}/v1/health`, {
      headers: { accept: "application/json" },
      cache: "no-store",
      signal: AbortSignal.timeout(400),
    });
    return response.ok ? base : null;
  } catch {
    return null;
  }
}

/** Product default `[Y-2, Y]` (#131 / #144). */
export function defaultSeasonYears(active = DEFAULT_ACTIVE_SEASON): number[] {
  return [active - 2, active - 1, active];
}

export function stubSeasonWindow(active = DEFAULT_ACTIVE_SEASON): SeasonWindow {
  return defaultSeasonYears(active);
}

export function stubHealth(
  overrides: Partial<Health> = {},
  options: Pick<ApiClientOptions, "stubCurrentSeasonMissing"> = {},
): Health {
  const missing = Boolean(options.stubCurrentSeasonMissing);
  const window = stubSeasonWindow();
  return {
    as_of: STUB_AS_OF,
    active_season: DEFAULT_ACTIVE_SEASON,
    current_season_missing: missing,
    season_window: window,
    source: "local",
    seasons_present: missing ? window.filter((year) => year < DEFAULT_ACTIVE_SEASON) : window,
    current_season_missing_reason: missing ? "stub_qa" : null,
    ...overrides,
  };
}

export function stubSeasons(
  active = DEFAULT_ACTIVE_SEASON,
  options: Pick<ApiClientOptions, "stubCurrentSeasonMissing"> = {},
): SeasonsResponse {
  const missing = Boolean(options.stubCurrentSeasonMissing);
  const window = defaultSeasonYears(active);
  return {
    as_of: STUB_AS_OF,
    active_season: active,
    season_window: window,
    seasons_present: missing ? window.filter((year) => year < active) : window,
    current_season_missing: missing,
  };
}

export function stubCardsFeed(): FantasyCard[] {
  return stubCards as FantasyCard[];
}

function normalizeBaseUrl(baseUrl: string | null | undefined): string | null {
  const trimmed = String(baseUrl || "").trim().replace(/\/+$/, "");
  return trimmed || null;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

function yearsFromUnknown(value: unknown): number[] {
  return Array.isArray(value) ? value.map(Number).filter(Number.isFinite) : [];
}

/** Keep OpenAPI array `[Y-2, Y]`. Expand `{start,end}` only if a mock still sends it. */
export function parseSeasonWindow(
  value: unknown,
  active = DEFAULT_ACTIVE_SEASON,
): SeasonWindow {
  if (Array.isArray(value)) {
    const years = yearsFromUnknown(value);
    if (years.length) {
      return years;
    }
  }
  const raw = asRecord(value);
  const start = Number(raw.start ?? raw.min);
  const end = Number(raw.end ?? raw.max);
  if (Number.isFinite(start) && Number.isFinite(end) && end >= start) {
    const years: number[] = [];
    for (let year = start; year <= end; year += 1) {
      years.push(year);
    }
    return years;
  }
  return defaultSeasonYears(active);
}

function parseSource(value: unknown): ArtifactSource {
  const raw = String(value || "").trim().toLowerCase();
  if (raw === "remote" || raw === "local" || raw === "missing") {
    return raw;
  }
  return "missing";
}

function parseHealth(payload: unknown): Health {
  const raw = asRecord(payload);
  const active = Number(raw.active_season) || DEFAULT_ACTIVE_SEASON;
  const present = yearsFromUnknown(raw.seasons_present);
  return {
    as_of: String(raw.as_of ?? ""),
    active_season: active,
    current_season_missing: Boolean(raw.current_season_missing),
    season_window: parseSeasonWindow(raw.season_window, active),
    source: parseSource(raw.source),
    seasons_present: present,
    current_season_missing_reason:
      raw.current_season_missing_reason == null
        ? null
        : String(raw.current_season_missing_reason),
  };
}

function parseSeasons(payload: unknown): SeasonsResponse {
  const raw = asRecord(payload);
  const active = Number(raw.active_season) || DEFAULT_ACTIVE_SEASON;
  const window = parseSeasonWindow(raw.season_window, active);
  // Honest years only. Do not fall back to season_window (may include unpublished 2026).
  const present = yearsFromUnknown(raw.seasons_present);
  return {
    as_of: String(raw.as_of ?? ""),
    active_season: active,
    season_window: window,
    seasons_present: present,
    current_season_missing: Boolean(raw.current_season_missing),
  };
}

function parseRec(value: unknown): RecommendationType | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const rec = String(value).trim().toLowerCase();
  return (RECOMMENDATION_TYPES as readonly string[]).includes(rec)
    ? (rec as RecommendationType)
    : null;
}

function parseCardsResponse(payload: unknown, source: "api" | "stub"): CardsResponse {
  const raw = asRecord(payload);
  const cards = Array.isArray(raw.cards) ? (raw.cards as FantasyCard[]) : [];
  const seasonRaw = raw.season;
  return {
    schema_version: String(raw.schema_version || SCHEMA_VERSION),
    as_of: String(raw.as_of ?? ""),
    season:
      seasonRaw === undefined || seasonRaw === null || seasonRaw === ""
        ? null
        : Number(seasonRaw),
    rec: parseRec(raw.rec),
    current_season_missing: Boolean(raw.current_season_missing),
    cards,
    source,
  };
}

function filterStubCards(cards: FantasyCard[], query: CardQuery = {}): FantasyCard[] {
  return cards.filter((card) => {
    if (query.season !== undefined && Number(card.season) !== Number(query.season)) {
      return false;
    }
    if (query.rec) {
      const wanted = String(query.rec).trim().toLowerCase();
      if (wanted && String(card.recommendation_type || "").toLowerCase() !== wanted) {
        return false;
      }
    }
    return true;
  });
}

function stubCardsResponse(
  query: CardQuery,
  missing: boolean,
): CardsResponse {
  return {
    schema_version: SCHEMA_VERSION,
    as_of: STUB_AS_OF,
    season: query.season ?? null,
    rec: parseRec(query.rec),
    current_season_missing: missing,
    cards: filterStubCards(stubCardsFeed(), query),
    source: "stub",
  };
}

async function getJson(
  fetcher: typeof fetch,
  url: string,
): Promise<unknown> {
  const response = await fetcher(url, {
    headers: { accept: "application/json" },
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status}`);
  }
  return response.json();
}

function parsePlayers(payload: unknown): PlayerRecord[] {
  const raw = asRecord(payload);
  return Array.isArray(raw.players) ? (raw.players as PlayerRecord[]) : [];
}

function parsePlayer(payload: unknown): PlayerRecord | null {
  const raw = asRecord(payload);
  if (raw.player === null || raw.player === undefined) {
    return null;
  }
  return raw.player as PlayerRecord;
}

function honestyFromUnknown(
  payload: unknown,
  options: Pick<ApiClientOptions, "stubCurrentSeasonMissing"> = {},
): Omit<PlayersResponse, "players" | "season"> {
  const raw = asRecord(payload);
  const health = parseHealth(payload);
  const windowYears = yearsFromUnknown(raw.season_window);
  return {
    as_of: health.as_of,
    active_season: health.active_season,
    current_season_missing: health.current_season_missing,
    season_window: windowYears.length ? windowYears : defaultSeasonYears(health.active_season),
    seasons_present: yearsFromUnknown(raw.seasons_present),
    source: raw.source === "remote" || raw.source === "local" || raw.source === "missing" ? raw.source : undefined,
    current_season_missing_reason:
      raw.current_season_missing_reason == null
        ? options.stubCurrentSeasonMissing
          ? "stub"
          : null
        : String(raw.current_season_missing_reason),
  };
}

function stubPlayerHonesty(
  options: Pick<ApiClientOptions, "stubCurrentSeasonMissing"> = {},
): Omit<PlayersResponse, "players" | "season"> {
  const health = stubHealth({}, options);
  return {
    as_of: health.as_of,
    active_season: health.active_season,
    current_season_missing: health.current_season_missing,
    season_window: defaultSeasonYears(health.active_season),
    seasons_present: options.stubCurrentSeasonMissing ? [] : defaultSeasonYears(health.active_season),
    source: "missing",
    current_season_missing_reason: options.stubCurrentSeasonMissing ? "stub" : null,
  };
}

export type BosApiClient = {
  readonly baseUrl: string | null;
  readonly source: "api" | "stub";
  getHealth(): Promise<Health>;
  getCards(query?: CardQuery): Promise<CardsResponse>;
  getSeasons(): Promise<SeasonsResponse>;
  getPlayers(query?: PlayerQuery): Promise<PlayersResponse>;
  getPlayer(id: string, query?: PlayerQuery): Promise<PlayerResponse>;
};

export function createApiClient(options: ApiClientOptions = {}): BosApiClient {
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  const fetcher = options.fetch ?? fetch;
  const stubMissing = Boolean(options.stubCurrentSeasonMissing);

  if (!baseUrl) {
    return {
      baseUrl: null,
      source: "stub",
      async getHealth() {
        return stubHealth({}, { stubCurrentSeasonMissing: stubMissing });
      },
      async getCards(query: CardQuery = {}) {
        return stubCardsResponse(query, stubMissing);
      },
      async getSeasons() {
        return stubSeasons(DEFAULT_ACTIVE_SEASON, { stubCurrentSeasonMissing: stubMissing });
      },
      async getPlayers(query: PlayerQuery = {}) {
        return {
          ...stubPlayerHonesty({ stubCurrentSeasonMissing: stubMissing }),
          season: query.season ?? null,
          players: [],
        };
      },
      async getPlayer(_id: string, query: PlayerQuery = {}) {
        return {
          ...stubPlayerHonesty({ stubCurrentSeasonMissing: stubMissing }),
          season: query.season ?? null,
          player: null,
        };
      },
    };
  }

  return {
    baseUrl,
    source: "api",
    async getHealth() {
      return parseHealth(await getJson(fetcher, `${baseUrl}/v1/health`));
    },
    async getCards(query: CardQuery = {}) {
      const params = new URLSearchParams();
      if (query.season !== undefined) {
        params.set("season", String(query.season));
      }
      if (query.rec) {
        params.set("rec", String(query.rec));
      }
      const qs = params.toString();
      const url = `${baseUrl}/v1/cards${qs ? `?${qs}` : ""}`;
      return parseCardsResponse(await getJson(fetcher, url), "api");
    },
    async getSeasons() {
      return parseSeasons(await getJson(fetcher, `${baseUrl}/v1/seasons`));
    },
    async getPlayers(query: PlayerQuery = {}) {
      const params = new URLSearchParams();
      if (query.season !== undefined) {
        params.set("season", String(query.season));
      }
      const qs = params.toString();
      const url = `${baseUrl}/v1/players${qs ? `?${qs}` : ""}`;
      const payload = await getJson(fetcher, url);
      return {
        ...honestyFromUnknown(payload),
        season: query.season ?? null,
        players: parsePlayers(payload),
      };
    },
    async getPlayer(id: string, query: PlayerQuery = {}) {
      const params = new URLSearchParams();
      if (query.season !== undefined) {
        params.set("season", String(query.season));
      }
      const qs = params.toString();
      const url = `${baseUrl}/v1/players/${encodeURIComponent(id)}${qs ? `?${qs}` : ""}`;
      const payload = await getJson(fetcher, url);
      return {
        ...honestyFromUnknown(payload),
        season: query.season ?? null,
        player: parsePlayer(payload),
      };
    },
  };
}

export function isRecommendationType(value: string): value is RecommendationType {
  return (RECOMMENDATION_TYPES as readonly string[]).includes(value);
}
