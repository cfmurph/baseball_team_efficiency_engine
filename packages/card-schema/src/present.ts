import { EARLY_MODEL_BADGE, PROMPT_LINE } from "./copy.ts";
import {
  ALLOWED_WAR_SOURCES,
  EDGE_UNIT,
  LABEL_TONES,
  RANK_NOUNS,
  RECOMMENDATION_LABELS,
  type FantasyCard,
  type Health,
  type RecommendationLabel,
  type ShareCardView,
} from "./types.ts";

function asRecord(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object"
    ? (value as Record<string, unknown>)
    : {};
}

export function recommendationLabel(
  recommendationType: string | null | undefined,
): string {
  const key = String(recommendationType || "")
    .trim()
    .toLowerCase();
  if (key in RECOMMENDATION_LABELS) {
    return RECOMMENDATION_LABELS[key];
  }
  const text = String(recommendationType || "").trim();
  return text ? text.toUpperCase() : "START";
}

export function warSource(card: FantasyCard): string {
  const raw = String(asRecord(card.edge).war_source || "")
    .trim()
    .toLowerCase();
  return (ALLOWED_WAR_SOURCES as readonly string[]).includes(raw) ? raw : "";
}

export function isApprox(card: FantasyCard): boolean {
  if (warSource(card) === "approx") {
    return true;
  }
  const flag = asRecord(card.edge).is_approx;
  return flag === true || flag === "true" || flag === 1;
}

export function playerName(card: FantasyCard): string {
  const name = asRecord(card.player).name;
  if (name === undefined || name === null || name === "") {
    return "";
  }
  return String(name).trim();
}

export function cardHeadline(card: FantasyCard): string {
  const headline = asRecord(card.share).headline;
  if (headline !== undefined && headline !== null && String(headline).trim()) {
    return String(headline).trim();
  }
  return playerName(card);
}

export function cardSubtitle(card: FantasyCard): string {
  const subtitle = asRecord(card.share).subtitle;
  if (subtitle !== undefined && subtitle !== null && String(subtitle).trim()) {
    return String(subtitle).trim();
  }
  const player = asRecord(card.player);
  const parts = ["name", "position", "team"]
    .map((key) => player[key])
    .filter((value) => value !== undefined && value !== null && value !== "")
    .map((value) => String(value).trim());
  return parts.join(" · ");
}

function formatEdge(value: unknown): string | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const amount = Number(value);
  if (!Number.isFinite(amount)) {
    return null;
  }
  const formatted = String(amount);
  const sign = amount >= 0 && !formatted.startsWith("+") ? "+" : "";
  return `${sign}${formatted} ${EDGE_UNIT}`;
}

function confidenceValue(card: FantasyCard): unknown {
  const edge = asRecord(card.edge);
  if ("confidence" in edge) {
    return edge.confidence;
  }
  return (card as { confidence?: unknown }).confidence;
}

function formatConfidence(value: unknown): string | null {
  if (value === undefined || value === null || value === "") {
    return null;
  }
  const raw = Number(value);
  if (!Number.isFinite(raw)) {
    return null;
  }
  const pct = raw <= 1 ? raw * 100 : raw;
  return `${Math.round(pct)}% conf`;
}

/** Rewrite leftover "vs repl" / "vs replacement" to "edge" on any face copy. */
export function normalizeStatLine(text: string | null | undefined): string {
  if (text === undefined || text === null || text === "") {
    return "";
  }
  let out = String(text);
  if (!/(vs\s+repl)/i.test(out)) {
    return out.split(/\s+/).join(" ").trim();
  }
  out = out.replace(/vs\s+replacement/gi, EDGE_UNIT);
  out = out.replace(/vs\s+repl\b/gi, EDGE_UNIT);
  const cleaned = out.split(/\s+/).join(" ").trim();
  if (/(vs\s+repl)/i.test(cleaned)) {
    return "";
  }
  return cleaned;
}

export function cardStatLine(card: FantasyCard): string {
  const statLine = asRecord(card.share).stat_line;
  if (statLine !== undefined && statLine !== null && String(statLine).trim()) {
    return normalizeStatLine(String(statLine).trim());
  }
  const bits: string[] = [];
  const edgeLine = formatEdge(asRecord(card.edge).vs_replacement);
  if (edgeLine) {
    bits.push(edgeLine);
  }
  if (!isApprox(card)) {
    const confLine = formatConfidence(confidenceValue(card));
    if (confLine) {
      bits.push(confLine);
    }
  }
  return normalizeStatLine(bits.join(" · "));
}

export function cardReason(card: FantasyCard): string {
  if (card.reason === undefined || card.reason === null) {
    return "";
  }
  return normalizeStatLine(String(card.reason));
}

export function cardAsOf(card: FantasyCard): string {
  if (card.as_of_date === undefined || card.as_of_date === null || card.as_of_date === "") {
    return "";
  }
  return String(card.as_of_date).trim();
}

export function cardRankLine(card: FantasyCard): string {
  const rank = asRecord(card.rank).among_rec_type;
  if (rank === undefined || rank === null || rank === "") {
    return "";
  }
  const place = Number(rank);
  if (!Number.isInteger(place)) {
    return "";
  }
  const rec = String(card.recommendation_type || "")
    .trim()
    .toLowerCase();
  const noun = RANK_NOUNS[rec] || rec || "pick";
  return `#${place} ${noun} tonight`;
}

export function presentCard(card: FantasyCard): ShareCardView {
  const recType = String(card.recommendation_type || "")
    .trim()
    .toLowerCase();
  return {
    recommendation_type: recType,
    label: recommendationLabel(recType),
    headline: cardHeadline(card),
    subtitle: cardSubtitle(card),
    stat_line: normalizeStatLine(cardStatLine(card)),
    reason: normalizeStatLine(cardReason(card)),
    as_of_date: cardAsOf(card),
    rank_line: cardRankLine(card),
    early_model: isApprox(card),
    card_id: String(card.card_id || "").trim(),
    prompt: PROMPT_LINE,
    player_id: String(asRecord(card.player).player_id || "").trim(),
  };
}

export function presentCards(cards: FantasyCard[]): ShareCardView[] {
  return cards.map(presentCard);
}

export function cardsForLabel(
  views: ShareCardView[],
  label: string | null | undefined,
): ShareCardView[] {
  if (!label) {
    return [...views];
  }
  const wanted = String(label).trim().toUpperCase();
  if (wanted === "ALL" || wanted === "*") {
    return [...views];
  }
  return views.filter((view) => view.label === wanted);
}

export function shareBlurb(view: ShareCardView): string {
  const identity = view.subtitle || view.headline;
  const lines = identity ? [`${view.label} — ${identity}`] : [view.label];
  if (
    view.headline &&
    identity &&
    view.headline !== identity &&
    !identity.includes(view.headline)
  ) {
    lines.push(view.headline);
  }
  const stat = normalizeStatLine(view.stat_line);
  if (stat) {
    lines.push(stat);
  }
  const reason = normalizeStatLine(view.reason);
  if (reason) {
    lines.push(reason);
  }
  if (view.as_of_date) {
    lines.push(`as of ${view.as_of_date}`);
  }
  return lines.join("\n");
}

export function cardShareFilename(
  view: ShareCardView,
  ext = "png",
): string {
  const raw = view.headline || view.subtitle || view.label;
  const slug =
    raw
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "card";
  const rec = (view.recommendation_type || view.label).toLowerCase().replace(/ /g, "-");
  return `benchorstart-${slug}-${rec}.${ext}`;
}

export function labelTone(label: string): string {
  return LABEL_TONES[label] || "#58a6ff";
}

export function shouldShowSeasonBanner(
  health: Health,
  seasons: number[] = [],
): boolean {
  if (health.current_season_missing) {
    return true;
  }
  const years = seasons.filter((year) => Number.isFinite(year));
  if (!years.length || !Number.isFinite(health.active_season)) {
    return false;
  }
  return Math.max(...years) < health.active_season;
}

export function earlyModelBadge(): string {
  return EARLY_MODEL_BADGE;
}
