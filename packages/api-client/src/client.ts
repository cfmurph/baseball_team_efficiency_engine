import {
  DEFAULT_ACTIVE_SEASON,
  RECOMMENDATION_TYPES,
  type CardsResponse,
  type FantasyCard,
  type Health,
  type RecommendationType,
  type SeasonWindow,
  type SeasonsResponse,
} from "@bos/card-schema";

import stubCards from "./stub-cards.json" with { type: "json" };

export type ApiClientOptions = {
  /** Base URL for the #106 service, e.g. `https://api.example.com`. No trailing slash. */
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

export function defaultSeasonYears(active = DEFAULT_ACTIVE_SEASON): number[] {
  return [active - 2, active - 1, active];
}

export function stubSeasonWindow(active = DEFAULT_ACTIVE_SEASON): SeasonWindow {
  return { start: active - 2, end: active };
}

export function stubHealth(
  overrides: Partial<Health> = {},
  options: Pick<ApiClientOptions, "stubCurrentSeasonMissing"> = {},
): Health {
  return {
    as_of: STUB_AS_OF,
    active_season: DEFAULT_ACTIVE_SEASON,
    current_season_missing: Boolean(options.stubCurrentSeasonMissing),
    season_window: stubSeasonWindow(),
    ...overrides,
  };
}

export function stubSeasons(active = DEFAULT_ACTIVE_SEASON): SeasonsResponse {
  return {
    seasons: defaultSeasonYears(active),
    active_season: active,
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

function parseHealth(payload: unknown): Health {
  const raw = asRecord(payload);
  const windowRaw = asRecord(raw.season_window);
  const windowList = Array.isArray(raw.season_window)
    ? (raw.season_window as unknown[])
    : [];
  const start =
    Number(windowRaw.start ?? windowRaw.min ?? windowList[0]) || DEFAULT_ACTIVE_SEASON - 2;
  const end =
    Number(windowRaw.end ?? windowRaw.max ?? windowList[1]) || DEFAULT_ACTIVE_SEASON;
  return {
    as_of: String(raw.as_of ?? raw.as_of_date ?? ""),
    active_season: Number(raw.active_season) || DEFAULT_ACTIVE_SEASON,
    current_season_missing: Boolean(raw.current_season_missing),
    season_window: { start, end },
  };
}

function yearsFromUnknown(value: unknown): number[] {
  return Array.isArray(value) ? value.map(Number).filter(Number.isFinite) : [];
}

function parseSeasons(payload: unknown): SeasonsResponse {
  if (Array.isArray(payload)) {
    const seasons = yearsFromUnknown(payload);
    return {
      seasons,
      active_season: seasons.length ? Math.max(...seasons) : DEFAULT_ACTIVE_SEASON,
    };
  }
  const raw = asRecord(payload);
  // #144 returns seasons_present (honest years) + season_window ([Y-2, Y]).
  // Keep `seasons` as a compat alias so older fixtures still parse.
  const named = yearsFromUnknown(raw.seasons);
  const present = yearsFromUnknown(raw.seasons_present);
  const seasons = named.length ? named : present;
  const windowYears = yearsFromUnknown(raw.season_window);
  const resolved = seasons.length ? seasons : windowYears;
  return {
    seasons: resolved,
    active_season: Number(raw.active_season) || (resolved.length ? Math.max(...resolved) : DEFAULT_ACTIVE_SEASON),
  };
}

function parseCards(payload: unknown): FantasyCard[] {
  if (Array.isArray(payload)) {
    return payload as FantasyCard[];
  }
  const raw = asRecord(payload);
  if (Array.isArray(raw.cards)) {
    return raw.cards as FantasyCard[];
  }
  if (Array.isArray(raw.items)) {
    return raw.items as FantasyCard[];
  }
  return [];
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
        return {
          cards: filterStubCards(stubCardsFeed(), query),
          source: "stub",
        };
      },
      async getSeasons() {
        return stubSeasons();
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
      return { cards: parseCards(await getJson(fetcher, url)), source: "api" };
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
