"use client";

import { useMemo, useState } from "react";

import {
  CURRENT_SEASON_BANNER,
  EMPTY_BODY,
  EMPTY_TAB,
  EMPTY_TITLE,
  FOOTER,
  HEADLINE,
  STUB_CAPTION,
  SUBHEAD,
  TAB_ALL,
  TAB_LABELS,
  cardsForLabel,
  type ShareCardView,
} from "@bos/card-schema";

import { ShareCard } from "@/components/ShareCard";
import { SiteHeader } from "@/components/SiteHeader";
import type { HomeData } from "@/lib/load";

const TABS = [TAB_ALL, ...TAB_LABELS] as const;

export function Home({
  views,
  showSeasonBanner,
  showingStubs,
}: HomeData) {
  const [tab, setTab] = useState<(typeof TABS)[number]>(TAB_ALL);
  const visible = useMemo(
    () => (tab === TAB_ALL ? views : cardsForLabel(views, tab)),
    [tab, views],
  );

  return (
    <div className="bos-shell">
      <SiteHeader active="cards" />

      {showSeasonBanner ? (
        <p className="bos-banner" role="status">
          {CURRENT_SEASON_BANNER}
        </p>
      ) : null}

      <section className="bos-hero">
        <h1>{HEADLINE}</h1>
        <p>{SUBHEAD}</p>
      </section>

      <section className="bos-feed" aria-label="This week's cards">
        {views.length === 0 ? (
          <div className="bos-empty" role="status">
            <h2>{EMPTY_TITLE}</h2>
            <p>{EMPTY_BODY}</p>
          </div>
        ) : (
          <>
            <div className="bos-tabs" role="tablist" aria-label="Recommendation">
              {TABS.map((label) => {
                const selected = tab === label;
                return (
                  <button
                    key={label}
                    type="button"
                    role="tab"
                    aria-selected={selected}
                    className={selected ? "bos-tab is-on" : "bos-tab"}
                    onClick={() => setTab(label)}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
            {visible.length === 0 ? (
              <p className="bos-caption">{EMPTY_TAB}</p>
            ) : (
              <ul className="bos-grid">
                {visible.map((view: ShareCardView, index) => (
                  <li key={view.card_id || `${view.label}-${index}`}>
                    <ShareCard view={view} featured={index === 0} />
                  </li>
                ))}
              </ul>
            )}
            {showingStubs ? <p className="bos-caption">{STUB_CAPTION}</p> : null}
          </>
        )}
      </section>

      <footer className="bos-foot">{FOOTER}</footer>
    </div>
  );
}
