/** Schema 1.0 BenchOrStart card + /v1 health types. */

export const SCHEMA_VERSION = "1.0";
export const EDGE_UNIT = "edge";
export const ALLOWED_WAR_SOURCES = ["bbref", "approx"] as const;
export const RECOMMENDATION_TYPES = ["start", "sit", "pickup", "stream"] as const;

export type WarSource = (typeof ALLOWED_WAR_SOURCES)[number];
export type RecommendationType = (typeof RECOMMENDATION_TYPES)[number];
export type RecommendationLabel = "START" | "BENCH" | "PICK UP" | "STREAM";

export type CardPlayer = {
  player_id?: string;
  name?: string;
  position?: string;
  team?: string;
};

export type CardEdge = {
  vs_replacement?: number | null;
  war?: number | null;
  war_source?: string | null;
  is_approx?: boolean | string | number | null;
  confidence?: number | null;
};

export type CardRank = {
  among_rec_type?: number | null;
};

export type CardShare = {
  headline?: string | null;
  subtitle?: string | null;
  stat_line?: string | null;
};

export type FantasyCard = {
  schema_version?: string;
  card_id?: string;
  recommendation_type?: string;
  as_of_date?: string;
  season?: number;
  player?: CardPlayer;
  edge?: CardEdge;
  rank?: CardRank;
  reason?: string;
  share?: CardShare;
};

export type ShareCardView = {
  recommendation_type: string;
  label: RecommendationLabel | string;
  headline: string;
  subtitle: string;
  stat_line: string;
  reason: string;
  as_of_date: string;
  rank_line: string;
  early_model: boolean;
  card_id: string;
  prompt: string;
};

export type SeasonWindow = {
  start: number;
  end: number;
};

export type Health = {
  as_of: string;
  active_season: number;
  current_season_missing: boolean;
  season_window: SeasonWindow;
};

export type SeasonsResponse = {
  seasons: number[];
  active_season: number;
};

export type CardsResponse = {
  cards: FantasyCard[];
  source: "api" | "stub";
};

export const LABEL_TONES: Record<string, string> = {
  START: "#3fb950",
  BENCH: "#f85149",
  "PICK UP": "#58a6ff",
  STREAM: "#d29922",
};

export const RECOMMENDATION_LABELS: Record<string, RecommendationLabel> = {
  start: "START",
  sit: "BENCH",
  pickup: "PICK UP",
  stream: "STREAM",
};

export const RANK_NOUNS: Record<string, string> = {
  start: "start",
  sit: "bench",
  pickup: "pickup",
  stream: "stream",
};

/** Locked default window [Y-2, Y] when Y = 2026 (#131). */
export const DEFAULT_ACTIVE_SEASON = 2026;
