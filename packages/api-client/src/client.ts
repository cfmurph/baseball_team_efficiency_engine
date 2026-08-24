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

export type BosApiClient = {
  readonly baseUrl: string | null;
  readonly source: "api" | "stub";
  getHealth(): Promise<Health>;
  getCards(query?: CardQuery): Promise<CardsResponse>;
  getSeasons(): Promise<SeasonsResponse>;
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
  };
}

export function isRecommendationType(value: string): value is RecommendationType {
  return (RECOMMENDATION_TYPES as readonly string[]).includes(value);
}
