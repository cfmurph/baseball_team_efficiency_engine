/** Local mock session only. Real login is #158 — no Clerk, Auth.js, or identity API. */

export const DEMO_USER_EMAIL = "demo@benchorstart.local";
export const DEMO_USER_LABEL = "Demo";
export const MOCK_SESSION_STORAGE_KEY = "bos_mock_session";
export const MOCK_SESSION_IN = "in";
export const MOCK_SESSION_OUT = "out";
export const MOCK_SESSION_EVENT = "bos-mock-session";

export type MockSession = {
  signedIn: boolean;
  email: string;
  label: string;
};

export type StorageLike = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
};

export function isMockSignedIn(stored: string | null | undefined): boolean {
  return stored !== MOCK_SESSION_OUT;
}

export function parseMockSession(stored: string | null | undefined): MockSession {
  const signedIn = isMockSignedIn(stored);
  return {
    signedIn,
    email: signedIn ? DEMO_USER_EMAIL : "",
    label: signedIn ? DEMO_USER_LABEL : "",
  };
}

function browserStorage(): StorageLike | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readMockSession(storage: StorageLike | null = browserStorage()): MockSession {
  if (!storage) {
    return parseMockSession(null);
  }
  try {
    return parseMockSession(storage.getItem(MOCK_SESSION_STORAGE_KEY));
  } catch {
    return parseMockSession(null);
  }
}

export function writeMockSession(
  signedIn: boolean,
  storage: StorageLike | null = browserStorage(),
): MockSession {
  const next = signedIn ? MOCK_SESSION_IN : MOCK_SESSION_OUT;
  if (storage) {
    try {
      storage.setItem(MOCK_SESSION_STORAGE_KEY, next);
    } catch {
      // Private mode / quota — still return the requested mock state.
    }
  }
  return parseMockSession(next);
}

export function notifyMockSessionChange(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new Event(MOCK_SESSION_EVENT));
}
