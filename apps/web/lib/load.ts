import {
  createApiClient,
  defaultDirectorySeason,
  listItemFromDetail,
  parsePlayerDetail,
  seasonWindowYears,
  type PlayerDetail,
  type PlayerListItem,
} from "@bos/api-client";
import {
  DEFAULT_ACTIVE_SEASON,
  presentCards,
  shouldShowSeasonBanner,
  type Health,
  type SeasonsResponse,
  type ShareCardView,
} from "@bos/card-schema";

import { parseCompareQuery, type CompareQuery, type CompareSearchInput } from "./compare.ts";

export type FeedSource = "api" | "stub";

export type HomeData = {
  views: ShareCardView[];
  health: Health;
  seasons: SeasonsResponse;
  source: FeedSource;
  showSeasonBanner: boolean;
  showingStubs: boolean;
};

function envFlag(name: string): boolean {
  return String(process.env[name] || "").trim().toLowerCase() === "true";
}

export function publicApiUrl(): string | null {
  const raw = String(process.env.NEXT_PUBLIC_API_URL || "").trim();
  return raw || null;
}

export async function loadHomeData(): Promise<HomeData> {
  const client = createApiClient({
    baseUrl: publicApiUrl(),
    stubCurrentSeasonMissing: envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING"),
  });

  try {
    const [health, seasons, feed] = await Promise.all([
      client.getHealth(),
      client.getSeasons(),
      client.getCards(),
    ]);
    const views = presentCards(feed.cards);
    return {
      views,
      health,
      seasons,
      source: feed.source,
      showSeasonBanner: shouldShowSeasonBanner(health, seasons.seasons),
      showingStubs: feed.source === "stub",
    };
  } catch {
    const health: Health = {
      as_of: "",
      active_season: DEFAULT_ACTIVE_SEASON,
      current_season_missing: true,
      season_window: { start: DEFAULT_ACTIVE_SEASON - 2, end: DEFAULT_ACTIVE_SEASON },
    };
    const seasons: SeasonsResponse = {
      seasons: [],
      active_season: DEFAULT_ACTIVE_SEASON,
    };
    return {
      views: [],
      health,
      seasons,
      source: "api",
      showSeasonBanner: true,
      showingStubs: false,
    };
  }
}

export type PlayersPageData = {
  bySeason: Record<number, PlayerListItem[]>;
  seasons: number[];
  defaultSeason: number;
  health: Health;
  showSeasonBanner: boolean;
  showingStubs: boolean;
};

export type PlayerPageData = {
  detail: PlayerDetail | null;
  seasons: number[];
  defaultSeason: number;
  health: Health;
  showSeasonBanner: boolean;
};

function missingHealth(): Health {
  return {
    as_of: "",
    active_season: DEFAULT_ACTIVE_SEASON,
    current_season_missing: true,
    season_window: { start: DEFAULT_ACTIVE_SEASON - 2, end: DEFAULT_ACTIVE_SEASON },
  };
}

export async function loadPlayersData(): Promise<PlayersPageData> {
  const client = createApiClient({
    baseUrl: publicApiUrl(),
    stubCurrentSeasonMissing: envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING"),
  });

  try {
    const [health, seasons] = await Promise.all([client.getHealth(), client.getSeasons()]);
    const years = seasonWindowYears(health);
    const defaultSeason = defaultDirectorySeason(health, seasons.seasons);
    const lists = await Promise.all(
      years.map((season) => client.getPlayers({ season })),
    );
    const bySeason: Record<number, PlayerListItem[]> = {};
    years.forEach((year, index) => {
      const source = client.source === "stub" ? "stub" : "api";
      bySeason[year] = (lists[index]?.players ?? [])
        .map((record) => {
          const detail = parsePlayerDetail({ player: record }, source);
          return detail ? listItemFromDetail(detail, year) : null;
        })
        .filter((row): row is PlayerListItem => row !== null);
    });
    return {
      bySeason,
      seasons: years,
      defaultSeason,
      health,
      showSeasonBanner: shouldShowSeasonBanner(health, seasons.seasons),
      showingStubs: client.source === "stub",
    };
  } catch {
    const health = missingHealth();
    return {
      bySeason: {},
      seasons: seasonWindowYears(health),
      defaultSeason: defaultDirectorySeason(health, []),
      health,
      showSeasonBanner: true,
      showingStubs: false,
    };
  }
}

export async function loadPlayerData(id: string): Promise<PlayerPageData> {
  const client = createApiClient({
    baseUrl: publicApiUrl(),
    stubCurrentSeasonMissing: envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING"),
  });

  try {
    const [health, seasons, response] = await Promise.all([
      client.getHealth(),
      client.getSeasons(),
      client.getPlayer(id),
    ]);
    const source = client.source === "stub" ? "stub" : "api";
    const detail = response.player ? parsePlayerDetail(response, source) : null;
    return {
      detail,
      seasons: seasonWindowYears(health),
      defaultSeason: defaultDirectorySeason(health, seasons.seasons),
      health,
      showSeasonBanner: shouldShowSeasonBanner(health, seasons.seasons),
    };
  } catch {
    const health = missingHealth();
    return {
      detail: null,
      seasons: seasonWindowYears(health),
      defaultSeason: defaultDirectorySeason(health, []),
      health,
      showSeasonBanner: true,
    };
  }
}

export type ComparePageData = {
  query: CompareQuery;
  details: Array<PlayerDetail | null>;
  bySeason: Record<number, PlayerListItem[]>;
  seasons: number[];
  defaultSeason: number;
  health: Health;
  showSeasonBanner: boolean;
  showingStubs: boolean;
};

export async function loadCompareData(raw: CompareSearchInput = {}): Promise<ComparePageData> {
  const directory = await loadPlayersData();
  const query = parseCompareQuery(raw, directory.defaultSeason, directory.seasons);
  if (!query.ids.length) {
    return {
      query,
      details: [],
      bySeason: directory.bySeason,
      seasons: directory.seasons,
      defaultSeason: directory.defaultSeason,
      health: directory.health,
      showSeasonBanner: directory.showSeasonBanner,
      showingStubs: directory.showingStubs,
    };
  }

  const client = createApiClient({
    baseUrl: publicApiUrl(),
    stubCurrentSeasonMissing: envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING"),
  });
  const source = client.source === "stub" ? "stub" : "api";
  const details = await Promise.all(
    query.ids.map(async (id) => {
      try {
        const response = await client.getPlayer(id);
        return response.player ? parsePlayerDetail(response, source) : null;
      } catch {
        return null;
      }
    }),
  );

  return {
    query,
    details,
    bySeason: directory.bySeason,
    seasons: directory.seasons,
    defaultSeason: directory.defaultSeason,
    health: directory.health,
    showSeasonBanner: directory.showSeasonBanner,
    showingStubs: directory.showingStubs,
  };
}
