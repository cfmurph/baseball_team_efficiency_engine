import Link from "next/link";

import { INVITE_CHIP, PRODUCT_NAME } from "@bos/card-schema";

type Props = {
  active: "cards" | "players" | "compare";
};

export function SiteHeader({ active }: Props) {
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
      </div>
    </header>
  );
}
