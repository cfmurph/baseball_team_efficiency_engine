"use client";

import Link from "next/link";
import { useCallback, useSyncExternalStore } from "react";

import { INVITE_CHIP, PRODUCT_NAME } from "@bos/card-schema";

import {
  DEMO_USER_EMAIL,
  DEMO_USER_LABEL,
  MOCK_SESSION_EVENT,
  isMockSignedIn,
  notifyMockSessionChange,
  readMockSession,
  writeMockSession,
} from "@/lib/mock-session";

type Props = {
  active: "cards" | "players" | "compare";
};

function subscribeMockSession(onStoreChange: () => void): () => void {
  if (typeof window === "undefined") {
    return () => {};
  }
  window.addEventListener(MOCK_SESSION_EVENT, onStoreChange);
  window.addEventListener("storage", onStoreChange);
  return () => {
    window.removeEventListener(MOCK_SESSION_EVENT, onStoreChange);
    window.removeEventListener("storage", onStoreChange);
  };
}

function mockSessionSnapshot(): boolean {
  return readMockSession().signedIn;
}

function mockSessionServerSnapshot(): boolean {
  return isMockSignedIn(null);
}

export function SiteHeader({ active }: Props) {
  const signedIn = useSyncExternalStore(
    subscribeMockSession,
    mockSessionSnapshot,
    mockSessionServerSnapshot,
  );

  const setSignedIn = useCallback((next: boolean) => {
    writeMockSession(next);
    notifyMockSessionChange();
  }, []);

  return (
    <header className="bos-top">
      <div className="bos-brandblock">
        <div className="bos-brandrow">
          <Link href="/" className="bos-brand">
            {PRODUCT_NAME}
          </Link>
          <span className="bos-chip">{INVITE_CHIP}</span>
        </div>
        <nav className="bos-nav" aria-label="Primary">
          <Link
            href="/"
            className={active === "cards" ? "bos-nav-link is-on" : "bos-nav-link"}
            aria-current={active === "cards" ? "page" : undefined}
          >
            Cards
          </Link>
          <Link
            href="/players"
            className={active === "players" ? "bos-nav-link is-on" : "bos-nav-link"}
            aria-current={active === "players" ? "page" : undefined}
          >
            Players
          </Link>
          <span className="bos-nav-link is-off" aria-disabled="true" title="Team compare comes next">
            Teams
          </span>
          <Link
            href="/compare"
            className={active === "compare" ? "bos-nav-link is-on" : "bos-nav-link"}
            aria-current={active === "compare" ? "page" : undefined}
          >
            Compare
          </Link>
        </nav>
        <div className="bos-session" data-signed-in={signedIn ? "true" : "false"}>
          {signedIn ? (
            <>
              <span className="bos-chip bos-session-demo">{DEMO_USER_LABEL}</span>
              <span className="bos-session-account" title="Local mock session — not a real account">
                {DEMO_USER_EMAIL}
              </span>
              <button type="button" className="bos-session-btn" onClick={() => setSignedIn(false)}>
                Log out
              </button>
            </>
          ) : (
            <button type="button" className="bos-session-btn" onClick={() => setSignedIn(true)}>
              Log in
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
