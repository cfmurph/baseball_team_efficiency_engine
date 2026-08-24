import {
  EARLY_MODEL_BADGE,
  PRODUCT_NAME,
  labelTone,
  normalizeStatLine,
  type ShareCardView,
} from "@bos/card-schema";

const WIDTH = 840;
const PAD = 44;
const BG = "#161b22";
const INK = "#e6edf3";
const MUTED = "#b1bac4";
const DIM = "#8b949e";
const WORDMARK = "#f85149";
const BADGE_INK = "#0d1117";
const EARLY = "#d29922";
const BORDER = "#30363d";

function wrap(text: string, width: number): string[] {
  if (!text) {
    return [];
  }
  const lines: string[] = [];
  for (const paragraph of text.split("\n")) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    let current = "";
    for (const word of words) {
      const next = current ? `${current} ${word}` : word;
      if (next.length > width && current) {
        lines.push(current);
        current = word;
      } else {
        current = next;
      }
    }
    lines.push(current || "");
  }
  return lines;
}

function measure(ctx: CanvasRenderingContext2D, text: string): number {
  return ctx.measureText(text).width;
}

function pill(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  text: string,
  fill: string,
  ink: string,
  stroke?: string,
): number {
  const padX = 14;
  const h = 30;
  const w = measure(ctx, text) + padX * 2;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, h / 2);
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.stroke();
  } else {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  ctx.fillStyle = ink;
  ctx.fillText(text, x + padX, y + 21);
  return w;
}

export function renderShareCardPng(view: ShareCardView): string {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("canvas");
  }

  const reasonLines = wrap(normalizeStatLine(view.reason), 52);
  const statLine = normalizeStatLine(view.stat_line);
  let contentH = 220 + (view.rank_line ? 28 : 0);
  contentH += view.headline ? 48 : 0;
  contentH += view.subtitle ? 28 : 0;
  contentH += statLine ? 32 : 0;
  contentH += 28 * reasonLines.length;
  contentH += view.as_of_date ? 36 : 0;
  const height = Math.max(420, contentH + 2 * PAD);

  canvas.width = WIDTH;
  canvas.height = height;

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, WIDTH, height);
  const tone = labelTone(view.label);
  ctx.fillStyle = tone;
  ctx.fillRect(0, 0, WIDTH, 8);
  ctx.fillStyle = BORDER;
  ctx.fillRect(0, 8, 2, height - 8);
  ctx.fillRect(WIDTH - 2, 8, 2, height - 8);
  ctx.fillRect(0, height - 2, WIDTH, 2);

  ctx.textBaseline = "alphabetic";
  let y = PAD + 20;
  const x = PAD;

  ctx.font = "700 16px Inter, system-ui, sans-serif";
  ctx.fillStyle = WORDMARK;
  ctx.fillText(PRODUCT_NAME.toUpperCase(), x, y);
  y += 36;

  ctx.font = "800 28px Inter, system-ui, sans-serif";
  ctx.fillStyle = INK;
  ctx.fillText(view.prompt, x, y);
  y += 28;

  ctx.font = "700 15px Inter, system-ui, sans-serif";
  const label = view.label || "START";
  const pillW = pill(ctx, x, y, label, tone, BADGE_INK);
  if (view.early_model) {
    pill(ctx, x + pillW + 12, y, EARLY_MODEL_BADGE.toUpperCase(), "transparent", EARLY, EARLY);
  }
  y += 52;

  ctx.font = "500 20px Inter, system-ui, sans-serif";
  if (view.rank_line) {
    ctx.fillStyle = DIM;
    ctx.fillText(view.rank_line, x, y);
    y += 28;
  }
  if (view.headline) {
    ctx.font = "800 36px Inter, system-ui, sans-serif";
    ctx.fillStyle = INK;
    ctx.fillText(view.headline, x, y);
    y += 48;
  }
  if (view.subtitle) {
    ctx.font = "500 20px Inter, system-ui, sans-serif";
    ctx.fillStyle = MUTED;
    ctx.fillText(view.subtitle, x, y);
    y += 30;
  }
  if (statLine) {
    ctx.font = "700 20px Inter, system-ui, sans-serif";
    ctx.fillStyle = INK;
    ctx.fillText(statLine, x, y);
    y += 32;
  }
  ctx.font = "400 20px Inter, system-ui, sans-serif";
  ctx.fillStyle = MUTED;
  for (const line of reasonLines) {
    ctx.fillText(line, x, y);
    y += 28;
  }
  if (view.as_of_date) {
    y += 8;
    ctx.font = "400 15px Inter, system-ui, sans-serif";
    ctx.fillStyle = DIM;
    ctx.fillText(`as of ${view.as_of_date}`, x, y);
  }

  return canvas.toDataURL("image/png");
}

export function downloadDataUrl(dataUrl: string, filename: string): void {
  const link = document.createElement("a");
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
}
