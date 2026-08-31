"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import {
  RECENT_GAME_LIMIT,
  formatCount,
  formatIp,
  hittingCountingLine,
  hittingRatesLine,
  isApproxWar,
  pitchingCountingLine,
  pitchingRatesLine,
  selectedYearMissing,
  type HittingSeason,
  type PitchingSeason,
  type PlayerSide,
} from "@bos/api-client";
import { CURRENT_SEASON_BANNER, EARLY_MODEL_BADGE, FOOTER, labelTone } from "@bos/card-schema";

import { SiteHeader } from "@/components/SiteHeader";
import { buildComparePath, compareHrefForPlayer } from "@/lib/compare";
import type { PlayerPageData } from "@/lib/load";

function lineForSeason<T extends { season: number }>(rows: T[], season: number): T | null {
  return rows.find((row) => row.season === season) || null;
}

function countingAndRates(side: PlayerSide, hitting: HittingSeason | null, pitching: PitchingSeason | null) {
  if (side === "pitching" && pitching) {
    return { counting: pitchingCountingLine(pitching), rates: pitchingRatesLine(pitching), approx: isApproxWar(pitching.war_source) };
  }
  if (hitting) {
    return { counting: hittingCountingLine(hitting), rates: hittingRatesLine(hitting), approx: isApproxWar(hitting.war_source) };
  }
  return { counting: "", rates: "", approx: false };
}

function AddToCompare({ playerId, season }: { playerId: string; season: number }) {
  const router = useRouter();
  const href = buildComparePath({ season, ids: [playerId] });
  return (
    <Link
      className="bos-ghost bos-add-compare"
      href={href}
      onClick={(event) => {
        const next = compareHrefForPlayer(playerId, season);
        if (next !== href) {
          event.preventDefault();
          router.push(next);
        }
      }}
    >
      Add to compare
    </Link>
  );
}

export function PlayerProfile({
  detail,
  seasons,
  defaultSeason,
  health,
}: PlayerPageData) {
  const [season, setSeason] = useState(defaultSeason);
  const [side, setSide] = useState<PlayerSide | null>(null);

  const hitting = detail ? lineForSeason(detail.hitting, season) : null;
  const pitching = detail ? lineForSeason(detail.pitching, season) : null;
  const hasBothSides = Boolean(hitting && pitching);

  const resolvedSide: PlayerSide = useMemo(() => {
    if (side) {
      return side;
    }
    if (pitching && !hitting) {
      return "pitching";
    }
    return "hitting";
  }, [hitting, pitching, side]);

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

  const activeRow = resolvedSide === "pitching" ? pitching : hitting;
  const lines = countingAndRates(resolvedSide, hitting, pitching);
  const yearMissing = selectedYearMissing(health, season, Boolean(activeRow));

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

      {yearMissing ? (
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
        <h1>
          {detail.player.name}
          {lines.approx ? <span className="bos-badge">{EARLY_MODEL_BADGE}</span> : null}
        </h1>
        <p>
          {detail.player.team} · {detail.player.position}
        </p>
        <p className="bos-identity-actions">
          <AddToCompare playerId={detail.player.player_id} season={season} />
        </p>
      </header>

      <div className="bos-tabs" role="tablist" aria-label="Season">
        {seasons.map((year) => (
          <button
            key={year}
            type="button"
            role="tab"
            aria-selected={season === year}
            className={season === year ? "bos-tab is-on" : "bos-tab"}
            onClick={() => {
              setSeason(year);
              setSide(null);
            }}
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

      {activeRow ? (
        <section className="bos-lines">
          {lines.counting ? <p className="bos-counting">{lines.counting}</p> : null}
          {lines.rates ? <p className="bos-rates">{lines.rates}</p> : null}
        </section>
      ) : (
        <div className="bos-empty" role="status">
          <h2>No {season} line yet</h2>
          <p>Empty tab until the nightly publishes that year. We do not invent rows.</p>
        </div>
      )}

      <section className="bos-block">
        <h2>Recent games</h2>
        {games.length === 0 ? (
          <p className="bos-caption">No recent games yet</p>
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
