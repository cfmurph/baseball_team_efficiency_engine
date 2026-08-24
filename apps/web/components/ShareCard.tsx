"use client";

import { useState } from "react";

import {
  COPIED,
  COPY_TEXT,
  DOWNLOAD_IMAGE,
  EARLY_MODEL_BADGE,
  PRODUCT_NAME,
  cardShareFilename,
  labelTone,
  normalizeStatLine,
  shareBlurb,
  type ShareCardView,
} from "@bos/card-schema";

import { downloadDataUrl, renderShareCardPng } from "@/lib/card-image";

type Props = {
  view: ShareCardView;
  featured?: boolean;
};

export function ShareCard({ view, featured = false }: Props) {
  const [copied, setCopied] = useState(false);
  const tone = labelTone(view.label);
  const blurb = shareBlurb(view);
  const statLine = normalizeStatLine(view.stat_line);
  const reason = normalizeStatLine(view.reason);

  async function onCopy() {
    try {
      await navigator.clipboard.writeText(blurb);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = blurb;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function onDownload() {
    const dataUrl = renderShareCardPng(view);
    downloadDataUrl(dataUrl, cardShareFilename(view));
  }

  return (
    <article
      className={featured ? "bos-card bos-card-featured" : "bos-card"}
      style={{ ["--bos-tone" as string]: tone }}
    >
      <div className="bos-wordmark">{PRODUCT_NAME}</div>
      <div className="bos-prompt">{view.prompt}</div>
      <div className="bos-pills">
        <span className="bos-label">{view.label}</span>
        {view.early_model ? (
          <span className="bos-badge">{EARLY_MODEL_BADGE}</span>
        ) : null}
      </div>
      {view.rank_line ? <div className="bos-rank">{view.rank_line}</div> : null}
      {view.headline ? <h2 className="bos-card-title">{view.headline}</h2> : null}
      {view.subtitle ? <div className="bos-sub">{view.subtitle}</div> : null}
      {statLine ? <div className="bos-stat">{statLine}</div> : null}
      {reason ? <p className="bos-reason">{reason}</p> : null}
      {view.as_of_date ? (
        <div className="bos-asof">as of {view.as_of_date}</div>
      ) : null}
      <div className="bos-actions">
        <button type="button" className="bos-ghost" onClick={onCopy}>
          {copied ? COPIED : COPY_TEXT}
        </button>
        <button type="button" className="bos-ghost" onClick={onDownload}>
          {DOWNLOAD_IMAGE}
        </button>
      </div>
    </article>
  );
}
