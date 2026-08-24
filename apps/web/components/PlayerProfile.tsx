"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import {
  RECENT_GAME_LIMIT,
  formatAvg,
  formatCount,
  formatEra,
  formatIp,
  formatOps,
  formatWar,
  formatWhip,
  formatWl,
  type HittingSeason,
  type PitchingSeason,
  type PlayerSide,
} from "@bos/api-client";
import { CURRENT_SEASON_BANNER, FOOTER, labelTone } from "@bos/card-schema";

import { SiteHeader } from "@/components/SiteHeader";
import type { PlayerPageData } from "@/lib/load";

function lineForSeason<T extends { season: number }>(rows: T[], season: number): T | null {
  return rows.find((row) => row.season === season) || null;
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bos-statbox">
      <span className="bos-statbox-value">{value}</span>
      <span className="bos-statbox-label">{label}</span>
    </div>
  );
}

function hittingHero(row: HittingSeason) {
  return [
    { label: "AVG", value: formatAvg(row.avg) },
    { label: "OBP", value: formatAvg(row.obp) },
    { label: "SLG", value: formatAvg(row.slg) },
    { label: "HR", value: formatCount(row.hr) },
    { label: "RBI", value: formatCount(row.rbi) },
    { label: "SB", value: formatCount(row.sb) },
  ];
}

function pitchingHero(row: PitchingSeason) {
  return [
    { label: "ERA", value: formatEra(row.era) },
    { label: "WHIP", value: formatWhip(row.whip) },
    { label: "IP", value: formatIp(row.ip) },
    { label: "K", value: formatCount(row.so) },
    { label: "W–L", value: formatWl(row) },
    { label: "WAR", value: formatWar(row.war) },
  ];
}

export function PlayerProfile({
  detail,
  seasons,
  defaultSeason,
  showSeasonBanner,
}: PlayerPageData) {
  const [season, setSeason] = useState(defaultSeason);
  const [side, setSide] = useState<PlayerSide | null>(null);

  const hasBothSides = Boolean(detail && detail.hitting.length && detail.pitching.length);

  const resolvedSide: PlayerSide = useMemo(() => {
    if (!detail) {
      return "hitting";
    }
    if (side) {
      return side;
    }
    if (detail.hitting.length && detail.pitching.length) {
      return detail.hitting.some((row) => row.season === season) ? "hitting" : "pitching";
    }
    return detail.pitching.length && !detail.hitting.length ? "pitching" : "hitting";
  }, [detail, season, side]);

  if (!detail) {
    return (
      <div className="bos-shell">
        <SiteHeader active="players" />
        <div className="bos-empty" role="status">
          <h2>Player not found</h2>
          <p>That id is not in the directory.</p>
          <p>
            <Link href="/players">Back to players</Link>
          </p>
        </div>
        <footer className="bos-foot">{FOOTER}</footer>
      </div>
    );
  }

  const hitting = lineForSeason(detail.hitting, season);
  const pitching = lineForSeason(detail.pitching, season);
  const activeRow = resolvedSide === "pitching" ? pitching : hitting;
  const hero = pitching && resolvedSide === "pitching"
    ? pitchingHero(pitching)
    : hitting
      ? hittingHero(hitting)
      : [];

  const hittingGames = detail.recent_games.hitting
    .filter((game) => !game.season || game.season === season)
    .slice(0, RECENT_GAME_LIMIT);
  const pitchingGames = detail.recent_games.pitching
    .filter((game) => !game.season || game.season === season)
    .slice(0, RECENT_GAME_LIMIT);
  const games = resolvedSide === "pitching" ? pitchingGames : hittingGames;

  return (
    <div className="bos-shell">
      <SiteHeader active="players" />

      {showSeasonBanner ? (
        <p className="bos-banner" role="status">
          {CURRENT_SEASON_BANNER}
        </p>
      ) : null}

      <p className="bos-crumb">
        <Link href="/players">Players</Link>
        <span aria-hidden="true"> / </span>
        <span>{detail.player.name}</span>
      </p>

      <header className="bos-identity">
        <h1>{detail.player.name}</h1>
        <p>
          {detail.player.position} · {detail.player.team} · {season}
        </p>
      </header>

      <div className="bos-chiprow" role="group" aria-label="Season">
        {seasons.map((year) => (
          <button
            key={year}
            type="button"
            className={season === year ? "bos-tab is-on" : "bos-tab"}
            aria-pressed={season === year}
            onClick={() => setSeason(year)}
          >
            {year}
          </button>
        ))}
      </div>

      {detail.card ? (
        <div className="bos-rec">
          <span
            className="bos-label"
            style={{ ["--bos-tone" as string]: labelTone(detail.card.label) }}
          >
            {detail.card.label}
          </span>
          {detail.card.reason ? <p>{detail.card.reason}</p> : null}
        </div>
      ) : null}

      {hasBothSides ? (
        <div className="bos-tabs" role="tablist" aria-label="Side">
          <button
            type="button"
            role="tab"
            aria-selected={resolvedSide === "hitting"}
            className={resolvedSide === "hitting" ? "bos-tab is-on" : "bos-tab"}
            onClick={() => setSide("hitting")}
          >
            Hitting
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={resolvedSide === "pitching"}
            className={resolvedSide === "pitching" ? "bos-tab is-on" : "bos-tab"}
            onClick={() => setSide("pitching")}
          >
            Pitching
          </button>
        </div>
      ) : null}

      {activeRow && hero.length ? (
        <div className="bos-hero-stats">
          {hero.map((stat) => (
            <Stat key={stat.label} label={stat.label} value={stat.value} />
          ))}
        </div>
      ) : (
        <div className="bos-empty" role="status">
          <h2>No {season} line yet</h2>
          <p>We do not invent a season that has not been published.</p>
        </div>
      )}

      {resolvedSide === "hitting" && detail.hitting.length ? (
        <section className="bos-block">
          <h2>Years</h2>
          <div className="bos-table-wrap">
            <table className="bos-table bos-table-compact">
              <thead>
                <tr>
                  <th>Year</th>
                  <th className="bos-num">G</th>
                  <th className="bos-num">PA</th>
                  <th className="bos-num">HR</th>
                  <th className="bos-num">RBI</th>
                  <th className="bos-num">SB</th>
                  <th className="bos-num">AVG</th>
                  <th className="bos-num">OPS</th>
                  <th className="bos-num">WAR</th>
                </tr>
              </thead>
              <tbody>
                {detail.hitting.map((row) => (
                  <tr key={row.season} className={row.season === season ? "is-on" : undefined}>
                    <td>{row.season}</td>
                    <td className="bos-num">{formatCount(row.g)}</td>
                    <td className="bos-num">{formatCount(row.pa)}</td>
                    <td className="bos-num">{formatCount(row.hr)}</td>
                    <td className="bos-num">{formatCount(row.rbi)}</td>
                    <td className="bos-num">{formatCount(row.sb)}</td>
                    <td className="bos-num">{formatAvg(row.avg)}</td>
                    <td className="bos-num">{formatOps(row.ops)}</td>
                    <td className="bos-num">{formatWar(row.war)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {resolvedSide === "pitching" && detail.pitching.length ? (
        <section className="bos-block">
          <h2>Years</h2>
          <div className="bos-table-wrap">
            <table className="bos-table bos-table-compact">
              <thead>
                <tr>
                  <th>Year</th>
                  <th className="bos-num">G</th>
                  <th className="bos-num">IP</th>
                  <th className="bos-num">W–L</th>
                  <th className="bos-num">K</th>
                  <th className="bos-num">ERA</th>
                  <th className="bos-num">WHIP</th>
                  <th className="bos-num">WAR</th>
                </tr>
              </thead>
              <tbody>
                {detail.pitching.map((row) => (
                  <tr key={row.season} className={row.season === season ? "is-on" : undefined}>
                    <td>{row.season}</td>
                    <td className="bos-num">{formatCount(row.g)}</td>
                    <td className="bos-num">{formatIp(row.ip)}</td>
                    <td className="bos-num">{formatWl(row)}</td>
                    <td className="bos-num">{formatCount(row.so)}</td>
                    <td className="bos-num">{formatEra(row.era)}</td>
                    <td className="bos-num">{formatWhip(row.whip)}</td>
                    <td className="bos-num">{formatWar(row.war)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      <section className="bos-block">
        <h2>Recent games</h2>
        {games.length === 0 ? (
          <p className="bos-caption">No recent games for {season}.</p>
        ) : resolvedSide === "pitching" ? (
          <div className="bos-table-wrap">
            <table className="bos-table bos-table-compact">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th className="bos-num">IP</th>
                  <th className="bos-num">H</th>
                  <th className="bos-num">ER</th>
                  <th className="bos-num">BB</th>
                  <th className="bos-num">K</th>
                  <th>Dec</th>
                </tr>
              </thead>
              <tbody>
                {pitchingGames.map((game) => (
                  <tr key={`${game.date}-${game.opponent}`}>
                    <td>{game.date}</td>
                    <td>{game.opponent}</td>
                    <td className="bos-num">{formatIp(game.ip)}</td>
                    <td className="bos-num">{formatCount(game.h)}</td>
                    <td className="bos-num">{formatCount(game.er)}</td>
                    <td className="bos-num">{formatCount(game.bb)}</td>
                    <td className="bos-num">{formatCount(game.so)}</td>
                    <td>{game.decision || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="bos-table-wrap">
            <table className="bos-table bos-table-compact">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Opp</th>
                  <th className="bos-num">AB</th>
                  <th className="bos-num">R</th>
                  <th className="bos-num">H</th>
                  <th className="bos-num">HR</th>
                  <th className="bos-num">RBI</th>
                  <th className="bos-num">BB</th>
                  <th className="bos-num">K</th>
                </tr>
              </thead>
              <tbody>
                {hittingGames.map((game) => (
                  <tr key={`${game.date}-${game.opponent}`}>
                    <td>{game.date}</td>
                    <td>{game.opponent}</td>
                    <td className="bos-num">{formatCount(game.ab)}</td>
                    <td className="bos-num">{formatCount(game.r)}</td>
                    <td className="bos-num">{formatCount(game.h)}</td>
                    <td className="bos-num">{formatCount(game.hr)}</td>
                    <td className="bos-num">{formatCount(game.rbi)}</td>
                    <td className="bos-num">{formatCount(game.bb)}</td>
                    <td className="bos-num">{formatCount(game.so)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <footer className="bos-foot">{FOOTER}</footer>
    </div>
  );
}
