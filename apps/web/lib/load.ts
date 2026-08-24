import { env } from "node:process";

import { createApiClient } from "@bos/api-client";
import {
  DEFAULT_ACTIVE_SEASON,
  presentCards,
  shouldShowSeasonBanner,
  type Health,
  type SeasonsResponse,
  type ShareCardView,
} from "@bos/card-schema";

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
  return String(env[name] || "").trim().toLowerCase() === "true";
}

export function publicApiUrl(): string | null {
  const raw = String(env.NEXT_PUBLIC_API_URL || "").trim();
  return raw || null;
}

export async function loadHomeData(): Promise<HomeData> {
  const client = createApiClient({
    baseUrl: publicApiUrl(),
    stubCurrentSeasonMissing:
      envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING") ||
      envFlag("STUB_CURRENT_SEASON_MISSING"),
  });

  try {
    const [health, seasons, feed] = await Promise.all([
      client.getHealth(),
      client.getSeasons(),
      client.getCards(),
    ]);
    const views = presentCards(feed.cards);
    const qaMissing =
      envFlag("NEXT_PUBLIC_STUB_CURRENT_SEASON_MISSING") ||
      envFlag("STUB_CURRENT_SEASON_MISSING");
    const present = health.seasons_present?.length
      ? health.seasons_present
      : (seasons.seasons_present ?? seasons.seasons);
    return {
      views,
      health,
      seasons,
      source: feed.source,
      showSeasonBanner: shouldShowSeasonBanner(health, present, {
        liveFeed: feed.source === "api" || qaMissing,
      }),
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
