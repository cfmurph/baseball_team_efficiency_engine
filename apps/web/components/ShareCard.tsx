"use client";

import { useState, type MouseEvent } from "react";

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

  async function onCopy(event: MouseEvent<HTMLButtonElement>) {
    const btn = event.currentTarget;
    btn.textContent = COPIED;
    setCopied(true);
    window.setTimeout(() => {
      btn.textContent = COPY_TEXT;
      setCopied(false);
    }, 4000);
    try {
      if (navigator.clipboard?.writeText) {
        await Promise.race([
          navigator.clipboard.writeText(blurb),
          new Promise((_, reject) =>
            window.setTimeout(() => reject(new Error("clipboard-timeout")), 400),
          ),
        ]);
        return;
      }
    } catch {
      // Permission prompt or missing API — fall through to execCommand.
    }
    const ta = document.createElement("textarea");
    ta.value = blurb;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch {
      // Clipboard is best-effort; the button still shows Copied.
    }
    ta.remove();
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
      {copied ? (
        <p className="bos-copied" role="status">
          {COPIED}
        </p>
      ) : null}
    </article>
  );
}
