"use client";

// Parked: Cole skipped waitlist. Home no longer mounts this as the CTA (#159).
// Streamlit fallback still uses fantasy/waitlist.py. Real login is #158.

import { FormEvent, useState } from "react";

import { CTA, EMAIL_ERROR, MICROCOPY, SUCCESS } from "@bos/card-schema";

type Status = "idle" | "ok" | "err";

export function WaitlistForm() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setPending(true);
    setStatus("idle");
    setError("");
    try {
      const response = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const payload = (await response.json()) as { ok?: boolean; error?: string };
      if (!response.ok || !payload.ok) {
        setStatus("err");
        setError(payload.error || EMAIL_ERROR);
        return;
      }
      setStatus("ok");
      setEmail("");
    } catch {
      setStatus("err");
      setError(EMAIL_ERROR);
    } finally {
      setPending(false);
    }
  }

  return (
    <section className="bos-waitlist" id="waitlist">
      <form onSubmit={onSubmit} className="bos-form">
        <label className="bos-sr" htmlFor="bos-email">
          Email
        </label>
        <input
          id="bos-email"
          name="email"
          type="email"
          inputMode="email"
          autoComplete="email"
          placeholder="you@email.com"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          required
        />
        <button type="submit" className="bos-cta" disabled={pending}>
          {CTA}
        </button>
      </form>
      {status === "ok" ? (
        <p className="bos-success" role="status">
          {SUCCESS}
        </p>
      ) : null}
      {status === "err" ? (
        <p className="bos-error" role="alert">
          {error}
        </p>
      ) : null}
      <p className="bos-micro">{MICROCOPY}</p>
    </section>
  );
}
