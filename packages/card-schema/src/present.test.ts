import assert from "node:assert/strict";
import test from "node:test";

import { EARLY_MODEL_BADGE, PROMPT_LINE } from "./copy.ts";
import {
  cardHeadline,
  cardReason,
  cardShareFilename,
  cardStatLine,
  cardsForLabel,
  isApprox,
  normalizeStatLine,
  presentCard,
  presentCards,
  recommendationLabel,
  shareBlurb,
  shouldShowSeasonBanner,
  warSource,
} from "./present.ts";
import type { FantasyCard, Health } from "./types.ts";

const steer: FantasyCard = {
  recommendation_type: "pickup",
  as_of_date: "2026-08-23",
  player: { name: "Spencer Steer", position: "1B", team: "CIN" },
  edge: {
    vs_replacement: 1.6,
    war_source: "bbref",
    is_approx: false,
    confidence: 0.81,
  },
  reason: "Grab him.",
  rank: { among_rec_type: 1 },
  share: {},
};

test("sit maps to BENCH and start stays START", () => {
  assert.equal(recommendationLabel("start"), "START");
  assert.equal(recommendationLabel("sit"), "BENCH");
  assert.equal(recommendationLabel("pickup"), "PICK UP");
  assert.equal(recommendationLabel("stream"), "STREAM");
});

test("empty headline falls back to player name, never the badge", () => {
  assert.equal(cardHeadline(steer), "Spencer Steer");
  assert.equal(cardHeadline({ ...steer, share: { headline: "   " } }), "Spencer Steer");
  assert.notEqual(cardHeadline(steer), recommendationLabel("pickup"));
});

test("stat line says edge, never vs repl, and hides approx confidence", () => {
  assert.equal(cardStatLine(steer), "+1.6 edge · 81% conf");
  assert.equal(normalizeStatLine("+1.6 vs repl"), "+1.6 edge");
  assert.equal(normalizeStatLine("1.6 vs replacement"), "1.6 edge");
  const approx: FantasyCard = {
    ...steer,
    recommendation_type: "sit",
    edge: {
      vs_replacement: -0.4,
      war_source: "approx",
      is_approx: true,
      confidence: 0.66,
    },
  };
  assert.equal(isApprox(approx), true);
  assert.equal(warSource(approx), "approx");
  assert.equal(cardStatLine(approx), "-0.4 edge");
  const view = presentCard(approx);
  assert.equal(view.label, "BENCH");
  assert.equal(view.early_model, true);
  assert.equal(view.prompt, PROMPT_LINE);
  assert.equal(view.early_model && EARLY_MODEL_BADGE, "early model");
  assert.equal(shareBlurb(view).includes("vs repl"), false);
});

test("share blurb is league-chat ready", () => {
  const view = presentCard({
    ...steer,
    reason: "Quiet week on the wire — grab him before your league chat does.",
  });
  const blurb = shareBlurb(view);
  assert.match(blurb, /^PICK UP — Spencer Steer/);
  assert.match(blurb, /\+1\.6 edge/);
  assert.equal(blurb.includes("vs repl"), false);
  assert.match(blurb, /Quiet week on the wire/);
  assert.match(blurb, /as of 2026-08-23/);
});

test("tabs filter by recommendation label", () => {
  const views = presentCards([
    { recommendation_type: "start" },
    { recommendation_type: "sit" },
    { recommendation_type: "pickup" },
  ]);
  assert.deepEqual(
    cardsForLabel(views, "START").map((view) => view.label),
    ["START"],
  );
  assert.deepEqual(
    cardsForLabel(views, "BENCH").map((view) => view.label),
    ["BENCH"],
  );
});

test("share filename slugs the player", () => {
  const view = presentCard(steer);
  assert.equal(cardShareFilename(view), "benchorstart-spencer-steer-pickup.png");
});

test("reason vs replacement is rewritten on face and copy, not the payload", () => {
  const dirty = "Aaron Judge is +3.4 vs replacement — lock this OF in.";
  const payload: FantasyCard = {
    ...steer,
    recommendation_type: "start",
    player: { name: "Aaron Judge", position: "OF", team: "NYY" },
    edge: {
      vs_replacement: 3.4,
      war_source: "bbref",
      is_approx: false,
      confidence: 0.91,
    },
    reason: dirty,
    share: { stat_line: "+3.4 edge · 91% conf" },
  };
  assert.equal(payload.reason, dirty);
  assert.equal(cardReason(payload), "Aaron Judge is +3.4 edge — lock this OF in.");
  const view = presentCard(payload);
  assert.equal(view.stat_line, "+3.4 edge · 91% conf");
  assert.equal(view.reason, "Aaron Judge is +3.4 edge — lock this OF in.");
  assert.equal(view.reason.toLowerCase().includes("vs repl"), false);
  const blurb = shareBlurb({
    ...view,
    reason: dirty,
  });
  assert.equal(blurb.toLowerCase().includes("vs repl"), false);
  assert.match(blurb, /Aaron Judge is \+3\.4 edge/);
  assert.equal(payload.reason, dirty);
});

test("season banner when missing or stale max year", () => {
  const base: Health = {
    as_of: "2025-09-01",
    active_season: 2026,
    current_season_missing: false,
    season_window: { start: 2024, end: 2026 },
  };
  assert.equal(shouldShowSeasonBanner(base, [2024, 2025, 2026]), false);
  assert.equal(
    shouldShowSeasonBanner({ ...base, current_season_missing: true }, [2024, 2025, 2026]),
    true,
  );
  assert.equal(shouldShowSeasonBanner(base, [2024, 2025]), true);
  assert.equal(shouldShowSeasonBanner(base, []), false);
  assert.equal(
    shouldShowSeasonBanner({ ...base, current_season_missing: true }, [2024], {
      liveFeed: false,
    }),
    false,
  );
  assert.equal(
    shouldShowSeasonBanner({ ...base, seasons_present: [2024, 2025] }, [2024, 2025, 2026]),
    true,
  );
});
