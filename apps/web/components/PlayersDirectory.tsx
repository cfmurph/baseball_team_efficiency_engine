"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { CURRENT_SEASON_BANNER, FOOTER } from "@bos/card-schema";
import {
  formatWar,
  type PlayerListItem,
} from "@bos/api-client";

import { SiteHeader } from "@/components/SiteHeader";
import type { PlayersPageData } from "@/lib/load";

function uniqueSorted(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))].sort((left, right) => left.localeCompare(right));
}

function Chip({
  label,
  selected,
  onClick,
}: {
  label: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={selected ? "bos-tab is-on" : "bos-tab"}
      aria-pressed={selected}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

export function PlayersDirectory({
  bySeason,
  seasons,
  defaultSeason,
  showSeasonBanner,
}: PlayersPageData) {
  const [query, setQuery] = useState("");
  const [season, setSeason] = useState(defaultSeason);
  const [position, setPosition] = useState("");
  const [team, setTeam] = useState("");

  const pool = bySeason[season] ?? [];
  const positions = useMemo(() => uniqueSorted(pool.map((row) => row.position)), [pool]);
  const teams = useMemo(() => uniqueSorted(pool.map((row) => row.team)), [pool]);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return pool.filter((row) => {
      if (position && row.position !== position) {
        return false;
      }
      if (team && row.team !== team) {
        return false;
      }
      if (!needle) {
        return true;
      }
      const hay = `${row.name} ${row.team} ${row.position} ${row.player_id}`.toLowerCase();
      return hay.includes(needle);
    });
  }, [pool, position, query, team]);

  return (
    <div className="bos-shell bos-shell-wide">
      <SiteHeader active="players" />

      {showSeasonBanner ? (
        <p className="bos-banner" role="status">
          {CURRENT_SEASON_BANNER}
        </p>
      ) : null}

      <section className="bos-pagehead">
        <h1>Players</h1>
        <p>Season lines for the live window, sorted by WAR.</p>
      </section>

      <div className="bos-filters">
        <label className="bos-search">
          <span className="bos-sr">Search players</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search name, team, or position"
          />
        </label>
        <div className="bos-chiprow" role="group" aria-label="Season">
          {seasons.map((year) => (
            <Chip
              key={year}
              label={String(year)}
              selected={season === year}
              onClick={() => {
                setSeason(year);
                setPosition("");
                setTeam("");
              }}
            />
          ))}
        </div>
        <div className="bos-chiprow" role="group" aria-label="Position">
          <Chip label="All pos" selected={!position} onClick={() => setPosition("")} />
          {positions.map((value) => (
            <Chip
              key={value}
              label={value}
              selected={position === value}
              onClick={() => setPosition(value)}
            />
          ))}
        </div>
        <div className="bos-chiprow" role="group" aria-label="Team">
          <Chip label="All teams" selected={!team} onClick={() => setTeam("")} />
          {teams.map((value) => (
            <Chip
              key={value}
              label={value}
              selected={team === value}
              onClick={() => setTeam(value)}
            />
          ))}
        </div>
      </div>

      {rows.length === 0 ? (
        <div className="bos-empty" role="status">
          <h2>No players in this slice</h2>
          <p>Try another season or clear the filters. We do not invent missing-year rows.</p>
        </div>
      ) : (
        <div className="bos-table-wrap">
          <table className="bos-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Pos</th>
                <th>Team</th>
                <th>Line</th>
                <th className="bos-num">WAR</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row: PlayerListItem) => (
                <tr key={`${row.player_id}-${row.season}`}>
                  <td>
                    <Link href={`/players/${encodeURIComponent(row.player_id)}`}>{row.name}</Link>
                  </td>
                  <td>{row.position}</td>
                  <td>{row.team}</td>
                  <td>{row.line}</td>
                  <td className="bos-num">{formatWar(row.war)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <footer className="bos-foot">{FOOTER}</footer>
    </div>
  );
}
