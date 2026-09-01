import assert from "node:assert/strict";
import test from "node:test";

import {
  DEMO_USER_EMAIL,
  DEMO_USER_LABEL,
  MOCK_SESSION_IN,
  MOCK_SESSION_OUT,
  MOCK_SESSION_STORAGE_KEY,
  isMockSignedIn,
  parseMockSession,
  readMockSession,
  writeMockSession,
} from "./mock-session.ts";

function memoryStorage(): {
  store: Map<string, string>;
  api: { getItem(key: string): string | null; setItem(key: string, value: string): void };
} {
  const store = new Map<string, string>();
  return {
    store,
    api: {
      getItem(key: string) {
        return store.get(key) ?? null;
      },
      setItem(key: string, value: string) {
        store.set(key, value);
      },
    },
  };
}

test("missing storage value defaults to the signed-in demo user", () => {
  assert.equal(isMockSignedIn(null), true);
  assert.equal(isMockSignedIn(undefined), true);
  assert.equal(isMockSignedIn(MOCK_SESSION_IN), true);
  assert.deepEqual(parseMockSession(null), {
    signedIn: true,
    email: DEMO_USER_EMAIL,
    label: DEMO_USER_LABEL,
  });
  assert.equal(DEMO_USER_EMAIL, "demo@benchorstart.local");
});

test("out flag is the only logged-out state", () => {
  assert.equal(isMockSignedIn(MOCK_SESSION_OUT), false);
  assert.deepEqual(parseMockSession(MOCK_SESSION_OUT), {
    signedIn: false,
    email: "",
    label: "",
  });
  assert.equal(isMockSignedIn("unexpected"), true);
});

test("read and write stay local — no identity payload", () => {
  const { store, api } = memoryStorage();
  assert.equal(readMockSession(api).signedIn, true);
  assert.equal(writeMockSession(false, api).signedIn, false);
  assert.equal(store.get(MOCK_SESSION_STORAGE_KEY), MOCK_SESSION_OUT);
  assert.equal(readMockSession(api).signedIn, false);
  assert.equal(writeMockSession(true, api).email, DEMO_USER_EMAIL);
  assert.equal(store.get(MOCK_SESSION_STORAGE_KEY), MOCK_SESSION_IN);
});
