import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { existsSync } from "node:fs";

import { EMAIL_ERROR } from "@bos/card-schema";

const EMAIL_RE = /^[^@\s]+@[^@\s]+\.[^@\s]+$/;
const WAITLIST_SOURCE = "benchorstart";

type WaitlistBody = {
  email?: unknown;
  source?: unknown;
  created_at?: unknown;
};

function normalizeEmail(value: unknown): string | null {
  const text = String(value ?? "").trim().toLowerCase();
  if (!text || !EMAIL_RE.test(text)) {
    return null;
  }
  return text;
}

function repoRoot(start = process.cwd()): string {
  let dir = start;
  for (let i = 0; i < 8; i += 1) {
    if (existsSync(path.join(dir, "fantasy", "copy.py"))) {
      return dir;
    }
    const parent = path.dirname(dir);
    if (parent === dir) {
      break;
    }
    dir = parent;
  }
  return start;
}

function waitlistPath(): string {
  const raw = String(process.env.FANTASY_WAITLIST_PATH || "").trim();
  if (raw) {
    return path.isAbsolute(raw) ? raw : path.join(repoRoot(), raw);
  }
  return path.join(repoRoot(), "data", "waitlist", "signups.jsonl");
}

async function appendSignup(email: string, createdAt: string): Promise<void> {
  const dest = waitlistPath();
  await mkdir(path.dirname(dest), { recursive: true });
  const record = JSON.stringify({
    email,
    source: WAITLIST_SOURCE,
    created_at: createdAt,
  });
  await appendFile(dest, `${record}\n`, "utf8");
}

async function postWebhook(email: string, createdAt: string): Promise<void> {
  const webhook = String(process.env.FANTASY_WAITLIST_WEBHOOK || "").trim();
  if (!webhook) {
    return;
  }
  const response = await fetch(webhook, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      email,
      source: WAITLIST_SOURCE,
      created_at: createdAt,
    }),
  });
  if (!response.ok) {
    throw new Error(`waitlist webhook ${response.status}`);
  }
}

export async function POST(request: Request): Promise<Response> {
  let payload: WaitlistBody = {};
  try {
    payload = (await request.json()) as WaitlistBody;
  } catch {
    payload = {};
  }
  const email = normalizeEmail(payload.email);
  if (!email) {
    return Response.json({ ok: false, error: EMAIL_ERROR }, { status: 400 });
  }
  const createdAt =
    typeof payload.created_at === "string" && payload.created_at.trim()
      ? payload.created_at.trim()
      : new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
  try {
    await postWebhook(email, createdAt);
    try {
      await appendSignup(email, createdAt);
    } catch {
      // Vercel / read-only FS: webhook (or a no-op) is enough for success.
    }
    return Response.json({ ok: true, email });
  } catch (error) {
    const message = error instanceof Error ? error.message : "waitlist failed";
    return Response.json({ ok: false, error: message }, { status: 502 });
  }
}
